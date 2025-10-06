from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
from pandas.tseries.offsets import MonthEnd, DateOffset
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from xgboost import XGBRegressor

def model():
    try:
        import cupy as cp
        HAS_CUPY = cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        cp = None
        HAS_CUPY = False

    BASE_DIR = Path(__file__).resolve().parent
    DATA_ROOT = BASE_DIR / "processed_data4"   # was BASE_DIR
    RET_DATA_PATH = BASE_DIR / "ret_parquet"
    FEATURE_LIST_PATH = BASE_DIR / "column_names.txt"
    TARGET = "stock_ret"
    ID_COLUMNS = ["gvkey", "iid", "excntry", "year", "month"]

    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"Parquet directory not found: {DATA_ROOT}")

    if not RET_DATA_PATH.exists():
        raise FileNotFoundError(f"Return parquet path not found: {RET_DATA_PATH}")

    raw_feature_names = [line.strip() for line in FEATURE_LIST_PATH.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(raw_feature_names)} candidate features.")

    dataset = ds.dataset(str(DATA_ROOT), format="parquet", partitioning="hive")
    schema = dataset.schema

    feature_names = []
    dropped_features = []
    for name in raw_feature_names:
        if name in ID_COLUMNS or name == TARGET:
            dropped_features.append(name)
            continue
        if name not in schema.names:
            dropped_features.append(name)
            continue
        field_type = schema.field(name).type
        if pa.types.is_floating(field_type) or pa.types.is_integer(field_type):
            feature_names.append(name)
        else:
            dropped_features.append(name)

    if dropped_features:
        print(f"Excluded non-numeric or redundant features: {sorted(set(dropped_features))}")
    print(f"Using {len(feature_names)} numeric features.")
    if HAS_CUPY:
        print("CuPy detected: using GPU-backed arrays for XGBoost.")
    else:
        print("CuPy unavailable; XGBoost will copy NumPy arrays to the GPU.")


    # %%
    feature_columns = feature_names + ID_COLUMNS
    feature_table = dataset.to_table(columns=feature_columns)
    features_df = feature_table.to_pandas()

    if RET_DATA_PATH.is_dir():
        returns_dataset = ds.dataset(str(RET_DATA_PATH), format="parquet", partitioning="hive")
    else:
        returns_dataset = ds.dataset(str(RET_DATA_PATH), format="parquet")

    returns_table = returns_dataset.to_table(columns=ID_COLUMNS + [TARGET])
    returns_df = returns_table.to_pandas()

    for df in (features_df, returns_df):
        df['year'] = df['year'].astype(int)
        df['month'] = df['month'].astype(int)
        df['gvkey'] = df['gvkey'].astype(str).str.strip()
        df['iid'] = df['iid'].astype(str).str.strip()
        df['excntry'] = df['excntry'].astype(str).str.strip()

    panel_df = features_df.merge(returns_df, on=ID_COLUMNS, how='inner')
    rows_dropped = len(features_df) - len(panel_df)
    if rows_dropped:
        print(f"Warning: Dropped {rows_dropped} feature rows when merging with returns.")

    panel_df['obs_date'] = pd.to_datetime({'year': panel_df['year'], 'month': panel_df['month'], 'day': 1}) + MonthEnd(0)
    panel_df = panel_df.sort_values(['obs_date', 'gvkey', 'iid']).reset_index(drop=True)

    panel_df[feature_names] = panel_df[feature_names].astype(np.float32)
    panel_df[TARGET] = panel_df[TARGET].astype(np.float32)

    n_months = panel_df['obs_date'].nunique()
    print(f"Panel rows: {len(panel_df):,}")
    print(f"Monthly slices: {n_months}")
    print(f"Date range: {panel_df['obs_date'].min().date()} to {panel_df['obs_date'].max().date()}")


    def _prepare(df: pd.DataFrame) -> pd.DataFrame:
        df = df.replace([np.inf, -np.inf], np.nan)
        return df.dropna(subset=[TARGET])

    panel_df = _prepare(panel_df)
    valid_features = [col for col in feature_names if not panel_df[col].isna().all()]
    removed_all_nan = sorted(set(feature_names) - set(valid_features))
    feature_names = valid_features
    print(f"After cleaning: {len(panel_df):,} rows across {panel_df['obs_date'].nunique()} months.")
    if removed_all_nan:
        print(f"Dropped all-NaN features: {removed_all_nan}")

    def _cross_sectional_scale(group: pd.DataFrame) -> pd.DataFrame:
        feats = group[feature_names]
        medians = feats.median(skipna=True)
        filled = feats.fillna(medians)
        ranks = filled.rank(method="dense") - 1.0
        max_rank = ranks.max()
        denom = max_rank.replace(0.0, np.nan)
        scaled = ranks.divide(denom, axis=1) * 2.0 - 1.0
        scaled = scaled.fillna(0.0).clip(-1.0, 1.0).astype(np.float32)
        group = group.copy()
        group[feature_names] = scaled
        return group

    panel_df = panel_df.groupby('obs_date', group_keys=False).apply(_cross_sectional_scale).reset_index(drop=True)
    print('Applied cross-sectional median fill, ranking, and scaling per month.')


    TRAIN_START = pd.Timestamp('2005-01-01')
    TRAIN_YEARS = 8
    VAL_YEARS = 2
    TEST_YEARS = 1

    OUTPUT_ROOT = Path('oos_preds_pca')
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


    PCA_VARIANCE = 0.95

    PARAM_GRID = [
        {"max_depth": 4, "min_child_weight": 5, "subsample": 0.90, "colsample_bytree": 0.80, "reg_lambda": 1.0, "gamma": 0.0},
        {"max_depth": 5, "min_child_weight": 10, "subsample": 0.85, "colsample_bytree": 0.75, "reg_lambda": 1.5, "gamma": 0.0},
        {"max_depth": 6, "min_child_weight": 15, "subsample": 0.80, "colsample_bytree": 0.70, "reg_lambda": 2.0, "gamma": 0.1},
        {"max_depth": 3, "min_child_weight": 20, "subsample": 0.95, "colsample_bytree": 0.85, "reg_lambda": 1.0, "gamma": 0.0},
    ]

    RIDGE_ALPHAS = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 0.5, 1.0, 5.0]

    max_date = panel_df['obs_date'].max()
    predictions_by_month = []
    window_summaries = []

    counter = 0
    while True:
        train_start = TRAIN_START
        train_end = train_start + DateOffset(years=TRAIN_YEARS + counter)
        val_start = train_end
        val_end = val_start + DateOffset(years=VAL_YEARS)
        test_start = val_end
        test_end = test_start + DateOffset(years=TEST_YEARS)

        if val_start > max_date or test_start > max_date:
            break

        mask_train = (panel_df['obs_date'] >= train_start) & (panel_df['obs_date'] < train_end)
        mask_val = (panel_df['obs_date'] >= val_start) & (panel_df['obs_date'] < val_end)
        mask_test = (panel_df['obs_date'] >= test_start) & (panel_df['obs_date'] < test_end)

        if mask_test.sum() == 0:
            break
        if mask_train.sum() == 0 or mask_val.sum() == 0:
            counter += 1
            continue

        X_train = panel_df.loc[mask_train, feature_names]
        y_train = panel_df.loc[mask_train, TARGET]
        X_val = panel_df.loc[mask_val, feature_names]
        y_val = panel_df.loc[mask_val, TARGET]
        X_test = panel_df.loc[mask_test, feature_names]
        y_test = panel_df.loc[mask_test, TARGET]

        feature_scaler = StandardScaler()
        X_train_scaled = feature_scaler.fit_transform(X_train)
        X_val_scaled = feature_scaler.transform(X_val)
        X_test_scaled = feature_scaler.transform(X_test)

        pca = PCA(n_components=PCA_VARIANCE, svd_solver='full')
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_val_pca = pca.transform(X_val_scaled)
        X_test_pca = pca.transform(X_test_scaled)

        component_scaler = StandardScaler()
        X_train_ready = component_scaler.fit_transform(X_train_pca)
        X_val_ready = component_scaler.transform(X_val_pca)
        X_test_ready = component_scaler.transform(X_test_pca)

        X_train_ready = X_train_ready.astype(np.float32)
        X_val_ready = X_val_ready.astype(np.float32)
        X_test_ready = X_test_ready.astype(np.float32)



        y_train_np = y_train.to_numpy(dtype=np.float32)
        y_val_np = y_val.to_numpy(dtype=np.float32)
        y_test_np = y_test.to_numpy(dtype=np.float32)

        if HAS_CUPY:
            X_train_xgb = cp.asarray(X_train_ready)
            X_val_xgb = cp.asarray(X_val_ready)
            X_test_xgb = cp.asarray(X_test_ready)
            y_train_xgb = cp.asarray(y_train_np)
            y_val_xgb = cp.asarray(y_val_np)
            y_test_xgb = cp.asarray(y_test_np)
        else:
            X_train_xgb = X_train_ready
            X_val_xgb = X_val_ready
            X_test_xgb = X_test_ready
            y_train_xgb = y_train_np
            y_val_xgb = y_val_np
            y_test_xgb = y_test_np

        pca_components = int(pca.n_components_)
        pca_var = float(pca.explained_variance_ratio_.sum())

        base_params = dict(
            objective='reg:squarederror',
            n_estimators=2000,
            learning_rate=0.05,
            device="cuda",
            tree_method='hist',
            missing=np.nan,
            n_jobs=-1,
            random_state=42,
            eval_metric='rmse',
            early_stopping_rounds=75,
        )

        best_model = None
        best_params = None
        best_val_rmse = float('inf')
        for trial in PARAM_GRID:
            trial_params = dict(base_params)
            trial_params.update(trial)
            trial_model = XGBRegressor(**trial_params)
            trial_model.fit(X_train_xgb, y_train_xgb, eval_set=[(X_val_xgb, y_val_xgb)], verbose=False)
            score = float(getattr(trial_model, 'best_score', np.inf))
            if score < best_val_rmse:
                best_val_rmse = score
                best_params = {k: (float(v) if isinstance(v, (np.floating, np.float32)) else v) for k, v in trial_params.items()}
                best_model = trial_model

        if best_model is None:
            counter += 1
            continue

        y_pred = best_model.predict(X_test_xgb)
        if HAS_CUPY and cp is not None and isinstance(y_pred, cp.ndarray):
            y_pred = cp.asnumpy(y_pred)
        errors = y_test_np - y_pred
        sse = float(np.square(errors).sum())
        tss_zero = float(np.square(y_test_np).sum())
        r2_oos = 1.0 - sse / tss_zero if tss_zero > 0 else float('nan')

        best_ridge_model = None
        best_ridge_alpha = None
        best_ridge_val_mse = float('inf')
        for alpha in RIDGE_ALPHAS:
            ridge = Ridge(alpha=alpha, fit_intercept=True, solver='svd')
            ridge.fit(X_train_ready, y_train)
            val_pred = ridge.predict(X_val_ready)
            mse = float(np.mean(np.square(y_val - val_pred)))
            if mse < best_ridge_val_mse:
                best_ridge_val_mse = mse
                best_ridge_alpha = float(alpha)
                best_ridge_model = ridge

        ridge_pred = best_ridge_model.predict(X_test_ready) if best_ridge_model is not None else np.zeros_like(y_pred)
        ridge_errors = y_test_np - ridge_pred
        ridge_sse = float(np.square(ridge_errors).sum())
        ridge_r2 = 1.0 - ridge_sse / tss_zero if tss_zero > 0 else float('nan')

        test_slice = panel_df.loc[mask_test, ID_COLUMNS + ['obs_date']].copy()
        test_slice['model_iteration'] = counter
        test_slice['predicted_stock_ret'] = y_pred.astype(np.float32)
        test_slice['actual_stock_ret'] = y_test_np.astype(np.float32)
        test_slice['squared_error'] = np.square(errors).astype(np.float32)
        test_slice['ridge_stock_ret'] = ridge_pred.astype(np.float32)
        test_slice['ridge_squared_error'] = np.square(ridge_errors).astype(np.float32)
        predictions_by_month.append(test_slice)

        for (year_val, month_val), df_month in test_slice.groupby(['year', 'month'], sort=False):
            year_int = int(year_val)
            month_int = int(month_val)
            dest = OUTPUT_ROOT / f"year={year_int}" / f"month={month_int}"
            dest.mkdir(parents=True, exist_ok=True)
            df_month.to_parquet(dest / 'part-0.parquet', index=False)

        train_months = panel_df.loc[mask_train, 'obs_date'].nunique()
        val_months = panel_df.loc[mask_val, 'obs_date'].nunique()
        test_months = panel_df.loc[mask_test, 'obs_date'].nunique()

        window_summaries.append({
            'iteration': counter,
            'train_rows': int(mask_train.sum()),
            'val_rows': int(mask_val.sum()),
            'test_rows': int(mask_test.sum()),
            'train_months': int(train_months),
            'val_months': int(val_months),
            'test_months': int(test_months),
            'train_end': (train_end - MonthEnd(1)).date(),
            'val_end': (val_end - MonthEnd(1)).date(),
            'test_start': test_start.date(),
            'test_end': (test_end - MonthEnd(1)).date(),
            'pca_components': pca_components,
            'pca_variance': pca_var,
            'xgb_best_iteration': int(best_model.best_iteration) if getattr(best_model, 'best_iteration', None) is not None else None,
            'xgb_val_rmse': best_val_rmse,
            'xgb_params': best_params,
            'xgb_oos_r2': r2_oos,
            'ridge_alpha': best_ridge_alpha,
            'ridge_val_mse': best_ridge_val_mse,
            'ridge_oos_r2': ridge_r2,
            'xgb_sse': sse,
            'ridge_sse': ridge_sse,
            'tss_zero': tss_zero,
        })

        counter += 1

    window_summary_df = pd.DataFrame(window_summaries)
    print(f"Completed {len(window_summary_df)} expanding-window fits.")
    window_summary_df

    if predictions_by_month:
        oos_predictions = pd.concat(predictions_by_month, ignore_index=True)
        oos_predictions['actual_squared'] = np.square(oos_predictions['actual_stock_ret']).astype(np.float32)
        overall_sse = float(oos_predictions['squared_error'].sum())
        overall_ridge_sse = float(oos_predictions['ridge_squared_error'].sum())
        overall_tss_zero = float(oos_predictions['actual_squared'].sum())
        overall_r2 = 1.0 - overall_sse / overall_tss_zero if overall_tss_zero > 0 else float('nan')
        overall_ridge_r2 = 1.0 - overall_ridge_sse / overall_tss_zero if overall_tss_zero > 0 else float('nan')
        print(f"OOS coverage: {oos_predictions['obs_date'].min().date()} to {oos_predictions['obs_date'].max().date()}")
        print(f"Total test observations: {len(oos_predictions):,}")
        print(f"Overall XGBoost OOS R^2 (Gu et al. 2020): {overall_r2:.10f}")
        print(f"Overall Ridge OOS R^2 (Gu et al. 2020): {overall_ridge_r2:.10f}")

        monthly_stats = oos_predictions.groupby('obs_date', as_index=False).agg(
            sse=('squared_error', 'sum'),
            ridge_sse=('ridge_squared_error', 'sum'),
            tss_zero=('actual_squared', 'sum'),
        )
        numer = monthly_stats['sse'].to_numpy(dtype=np.float64)
        numer_ridge = monthly_stats['ridge_sse'].to_numpy(dtype=np.float64)
        denom = monthly_stats['tss_zero'].to_numpy(dtype=np.float64)
        monthly_stats['xgb_oos_r2'] = 1.0 - np.divide(
            numer,
            denom,
            out=np.full(numer.shape, np.nan, dtype=np.float64),
            where=denom > 0,
        )
        monthly_stats['ridge_oos_r2'] = 1.0 - np.divide(
            numer_ridge,
            denom,
            out=np.full(numer_ridge.shape, np.nan, dtype=np.float64),
            where=denom > 0,
        )

        monthly_r2 = monthly_stats[['obs_date', 'xgb_oos_r2', 'ridge_oos_r2']]
        print('First five monthly OOS R^2 values:')
        print(monthly_r2.head())
        print('Worst five XGBoost months:')
        print(monthly_r2.nsmallest(5, 'xgb_oos_r2'))
        print('Worst five Ridge months:')
        print(monthly_r2.nsmallest(5, 'ridge_oos_r2'))
        print('Sample predictions:')
        oos_predictions.head()

        print(f"Predictions written to {OUTPUT_ROOT.resolve()}")

        if 'window_summary_df' in globals() and not window_summary_df.empty:
            print('PCA component summary (first five windows):')
            print(window_summary_df[['iteration', 'pca_components', 'pca_variance']].head())
            avg_components = window_summary_df['pca_components'].mean()
            avg_var = window_summary_df['pca_variance'].mean()
            print(f'Average components retained: {avg_components:.2f}, average variance captured: {avg_var:.4f}')
    else:
        print('No out-of-sample predictions were generated.')

if __name__ == "__main__":
    model()