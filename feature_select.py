import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Iterable
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from collections import defaultdict

# Optional: pip install lightgbm
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

INPUT_ROOT = Path("processed_data4")
ID_COLS = ["gvkey", "iid", "excntry"]
PARTITION_COLS = ["year", "month"]
TARGET = "stock_ret"

# ---- Tunables ----
ABS_R_THRESHOLD = 0.90            # redundancy cut
LOW_PCT, HIGH_PCT = 0.01, 0.99    # cross-sectional winsorization
MAX_AFTER_IC = 60                 # cap after predictive screen
EMBEDDED_KEEP = 40                # final keep count (e.g., 30-40)
N_CV_SPLITS = 5                   # blocked CV splits (by month order)
SEED = 2025
USE_PERMUTATION = False           # if True, compute permutation importance instead of gain
RUN_STABILITY = True              # bootstrapped stability selection
N_BOOTSTRAPS = 20
STABILITY_TOPK = 40
STABILITY_KEEP_AT_LEAST = 0.50    # appear in top-k in >= 50% of bootstraps

# ---------- Utilities ----------
def load_panel_months(root: Path) -> List[Tuple[int, int]]:
    months = []
    for y_dir in sorted(root.glob("y=*")):
        y = int(y_dir.name.split("=")[1])
        for m_dir in sorted(y_dir.glob("m=*")):
            m = int(m_dir.name.split("=")[1])
            # require the parquet
            if (m_dir / "part-0.parquet").exists():
                months.append((y, m))
    months.sort()
    return months

def read_month(root: Path, y: int, m: int) -> pd.DataFrame:
    p = root / f"y={y}" / f"m={m}" / "part-0.parquet"
    df = pd.read_parquet(p)
    return df

def month_key(df: pd.DataFrame) -> pd.Series:
    return df["year"].astype(int) * 100 + df["month"].astype(int)

def next_month(year: int, month: int) -> Tuple[int, int]:
    if month == 12:
        return year + 1, 1
    return year, month + 1

def is_interpretable(name: str) -> Tuple[int, int]:
    # Lower tuple sorts “more interpretable” first:
    # 1) fewer non-alphanumeric chars  2) shorter length
    non_alnum = sum(not ch.isalnum() and ch != '_' for ch in name)
    return (non_alnum, len(name))

def winsorize_and_zscore(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    # Cross-sectional per (year, month) winsorize and z-score
    def _one_group(g: pd.DataFrame) -> pd.DataFrame:
        h = g.copy()
        ql = h[feature_cols].quantile(LOW_PCT)
        qh = h[feature_cols].quantile(HIGH_PCT)
        h[feature_cols] = h[feature_cols].clip(lower=ql, upper=qh, axis=1)
        mu = h[feature_cols].mean()
        sd = h[feature_cols].std(ddof=0).replace(0.0, np.nan)
        h[feature_cols] = (h[feature_cols] - mu) / sd
        return h
    parts = []
    for (_, _), g in df.groupby(["year", "month"], sort=False):
        parts.append(_one_group(g))
    return pd.concat(parts, ignore_index=True)

def cross_sectional_corr_abs(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    # Mean absolute correlation across months
    corr_sums = defaultdict(lambda: defaultdict(list))
    for (_, _), g in df.groupby(["year", "month"], sort=False):
        X = g[cols]
        if len(X) < 3:
            continue
        # Use Pearson on standardized features (we z-score already)
        C = X.corr(method="pearson")
        for i, ci in enumerate(cols):
            for j in range(i+1, len(cols)):
                cj = cols[j]
                r = abs(C.iloc[i, j])
                if pd.notna(r):
                    corr_sums[ci][cj].append(r)
    rows = []
    for ci, d in corr_sums.items():
        for cj, arr in d.items():
            rows.append((ci, cj, float(np.nanmean(arr)) if len(arr) else np.nan))
    out = pd.DataFrame(rows, columns=["f1", "f2", "mean_abs_corr"])
    return out

def redundancy_filter(df_std: pd.DataFrame, feature_cols: List[str], thr: float) -> List[str]:
    # Build an undirected graph of high-corr pairs aggregated across months
    corr_df = cross_sectional_corr_abs(df_std, feature_cols)
    high = corr_df[(corr_df["mean_abs_corr"] >= thr) & corr_df["mean_abs_corr"].notna()]
    keep = set(feature_cols)
    # Greedy removal: for each high-corr pair, drop the less interpretable if both still present
    for f1, f2, _ in high.sort_values("mean_abs_corr", ascending=False).itertuples(index=False):
        if f1 in keep and f2 in keep:
            drop = max([f1, f2], key=lambda n: is_interpretable(n))  # worse interpretability
            keep.remove(drop)
    return sorted(keep)

def make_next_month_returns(panel: pd.DataFrame) -> pd.DataFrame:
    # Create next-month realized return aligned to each row at month t
    # Assumes IDs (gvkey,iid,excntry) identify an asset cross-sectionally.
    next_rows = []
    for (y, m), g in panel.groupby(["year", "month"], sort=False):
        ny, nm = next_month(y, m)
        # Join t cross-section with t+1 returns
        try:
            g_next = panel[(panel["year"] == ny) & (panel["month"] == nm)][ID_COLS + [TARGET]].copy()
        except KeyError:
            continue
        g_next = g_next.rename(columns={TARGET: f"{TARGET}_t1"})
        h = g.merge(g_next, on=ID_COLS, how="left", validate="many_to_one")
        h["year"] = y
        h["month"] = m
        next_rows.append(h)
    out = pd.concat(next_rows, ignore_index=True) if next_rows else panel.copy()
    return out

from scipy.stats import spearmanr
import numpy as np

def low_variation_filter(panel: pd.DataFrame, cols: list[str], min_active_frac=0.6, tol=1e-8):
    stats = []
    for (_, _), g in panel.groupby(["year", "month"], sort=False):
        s = g[cols].std(ddof=0)
        # NaN -> False before comparison; avoids invalid-value warnings
        active = s.fillna(0.0) > tol
        stats.append(active)

    act = pd.DataFrame(stats).mean(axis=0)  # fraction of months with variation
    return act[act >= min_active_frac].index.tolist()



def rolling_rank_ic(panel_std: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    pnl = panel_std.dropna(subset=[f"{TARGET}_t1"])
    by_month = []
    for (y, m), g in pnl.groupby(["year", "month"], sort=False):
        y_true = g[f"{TARGET}_t1"]
        if y_true.notna().sum() < 10:
            continue

        # Precompute ranks once
        ranks = g[feature_cols].rank(axis=0, method="average", na_option="keep")

        month_res = {}
        for c in feature_cols:
            x = ranks[c]
            mask = x.notna() & y_true.notna()
            n = int(mask.sum())

            # guardrails: enough points, and both vectors have variation
            if n < 10:
                continue
            if x[mask].nunique(dropna=True) < 2:
                continue
            if y_true[mask].nunique(dropna=True) < 2:
                continue

            rho, _ = spearmanr(x[mask], y_true[mask])
            if np.isfinite(rho):
                month_res[c] = rho

        if month_res:
            by_month.append(pd.Series(month_res, name=f"{y}-{m:02d}"))

    ic_table = pd.DataFrame(by_month)
    med_ic = ic_table.median(axis=0, skipna=True)
    sign_stability = (np.sign(ic_table.fillna(0.0)) > 0).mean(axis=0)
    return pd.DataFrame({"median_ic": med_ic, "sign_stability": sign_stability}) \
             .sort_values("median_ic", ascending=False)


def blocked_month_splits(month_keys: List[int], n_splits: int) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    # Deterministic splits by ordered unique month integers (YYYYMM)
    uniq = sorted(sorted(set(month_keys)))
    if len(uniq) <= n_splits:
        n_splits = max(2, len(uniq) - 1)
    ts = TimeSeriesSplit(n_splits=n_splits)
    idx_map = {mk: i for i, mk in enumerate(uniq)}
    month_idx = np.array([idx_map[k] for k in month_keys])
    # Yield boolean masks based on month index folds
    for tr_idx, va_idx in ts.split(uniq):
        tr_mask = np.isin(month_idx, tr_idx)
        va_mask = np.isin(month_idx, va_idx)
        yield np.where(tr_mask)[0], np.where(va_mask)[0]


def train_lightgbm_embedded(panel_std: pd.DataFrame, feature_cols: list[str]) -> pd.Series:
    X = panel_std[feature_cols].values          # numpy on CPU is fine
    y = panel_std["stock_ret_t1"].values
    mk = (panel_std["year"].astype(int)*100 + panel_std["month"].astype(int)).tolist()

    params = dict(
    n_estimators=1500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    tree_method="hist",
    device="cuda",
    eval_metric="rmse",
    random_state=2025,
    early_stopping_rounds=100,   # <- move here
)




    oof = np.full(len(y), np.nan)
    models = []
    for tr, va in blocked_month_splits(mk, N_CV_SPLITS):
        m = XGBRegressor(**params)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
        models.append(m)
        oof[va] = m.predict(X[va])

    # Average normalized gain across folds
    gain = np.zeros(len(feature_cols), dtype=float)
    for m in models:
        score = m.get_booster().get_score(importance_type="gain")
        vec = np.array([score.get(f, 0.0) for f in feature_cols], dtype=float)
        s = vec.sum()
        if s > 0:
            vec /= s
        gain += vec
    gain /= max(1, len(models))

    rmse = float(np.sqrt(mean_squared_error(y[~np.isnan(oof)], oof[~np.isnan(oof)])))
    imp = pd.Series(gain, index=feature_cols, name="gain_importance").sort_values(ascending=False)
    imp.attrs["cv_rmse"] = rmse
    return imp


    
def permutation_importance_simple(panel_std: pd.DataFrame, feature_cols: List[str], n_rounds: int = 3) -> pd.Series:
    # Simple, model-agnostic permutation importance with blocked CV
    X = panel_std[feature_cols].copy()
    y = panel_std[f"{TARGET}_t1"].values
    mk = (panel_std["year"].astype(int) * 100 + panel_std["month"].astype(int)).tolist()

    rng = np.random.default_rng(SEED)
    base_preds = np.full(len(y), np.nan)

    # Use a single ref model per fold, then permute features within the fold
    fold_models = []
    fold_indices = []
    for tr, va in blocked_month_splits(mk, N_CV_SPLITS):
        dtr = lgb.Dataset(X.iloc[tr].values, label=y[tr], free_raw_data=False)
        dva = lgb.Dataset(X.iloc[va].values, label=y[va], reference=dtr, free_raw_data=False)
        bst = lgb.train(
            dict(objective="regression", metric="rmse", learning_rate=0.05, num_leaves=31, verbosity=-1, seed=SEED),
            dtr, num_boost_round=1000, valid_sets=[dtr, dva], early_stopping_rounds=50, verbose_eval=False
        )
        base_preds[va] = bst.predict(X.iloc[va].values, num_iteration=bst.best_iteration)
        fold_models.append(bst)
        fold_indices.append((tr, va))

    base_rmse = float(np.sqrt(mean_squared_error(y[~np.isnan(base_preds)], base_preds[~np.isnan(base_preds)])))
    drops = pd.Series(0.0, index=feature_cols)
    for (bst, (tr, va)) in zip(fold_models, fold_indices):
        Xv = X.iloc[va].copy()
        for c in feature_cols:
            delta = 0.0
            for _ in range(n_rounds):
                perm = Xv[c].to_numpy()
                rng.shuffle(perm)
                Xv[c] = perm
                pred = bst.predict(Xv.values, num_iteration=bst.best_iteration)
                rmse = np.sqrt(mean_squared_error(y[va], pred))
                delta += (rmse - base_rmse)
            drops[c] += delta / n_rounds
    drops /= len(fold_indices)
    return drops.sort_values(ascending=False).rename("perm_importance")

# ---------- Main pipeline ----------
@dataclass
class TrainWindow:
    start: Tuple[int, int]
    end: Tuple[int, int]  # inclusive

def between_months(y: int, m: int, start: Tuple[int, int], end: Tuple[int, int]) -> bool:
    key = y * 100 + m
    return (start[0]*100 + start[1]) <= key <= (end[0]*100 + end[1])

def run_pipeline(window: TrainWindow) -> Dict[str, List[str]]:
    months = load_panel_months(INPUT_ROOT)
    sel_months = [(y, m) for (y, m) in months if between_months(y, m, window.start, window.end)]
    if not sel_months:
        raise ValueError("No months in the requested training window.")

    # 0) Load panel
    dfs = [read_month(INPUT_ROOT, y, m) for (y, m) in sel_months]
    panel = pd.concat(dfs, ignore_index=True)
    # Limit to numeric features (exclude IDs/partitions/target)
    numeric = panel.select_dtypes(include="number").columns.tolist()
    feature_cols = [c for c in numeric if c not in ID_COLS + PARTITION_COLS + [TARGET]]
    # Ensure IDs exist
    for c in ID_COLS + PARTITION_COLS + [TARGET]:
        if c not in panel.columns:
            raise ValueError(f"Missing required column: {c}")

    # 1) Quick filters: standardize and redundancy
    panel_std = winsorize_and_zscore(panel[ID_COLS + PARTITION_COLS + [TARGET] + feature_cols].copy(), feature_cols)
    feature_cols = low_variation_filter(panel_std, feature_cols)
    kept_after_corr = redundancy_filter(panel_std, feature_cols, ABS_R_THRESHOLD)

    # 2) Predictive screen: rank IC vs next-month returns
    panel_next = make_next_month_returns(panel_std)
    ic_table = rolling_rank_ic(panel_next, kept_after_corr)
    ic_pass = ic_table[(ic_table["median_ic"] > 0.0) & (ic_table["sign_stability"] >= 0.60)].index.tolist()
    ic_ranked = ic_table.loc[ic_pass].sort_values("median_ic", ascending=False).index.tolist()
    kept_after_ic = ic_ranked[:MAX_AFTER_IC]

    # 3) Embedded selection: LightGBM importance (gain or permutation)
    use_cols = kept_after_ic
    panel_emb = panel_next.dropna(subset=[f"{TARGET}_t1"]).copy()
    panel_emb = panel_emb[ID_COLS + PARTITION_COLS + [f"{TARGET}_t1"] + use_cols]
    # Re-standardize features before model (safer for permutation)
    panel_emb = winsorize_and_zscore(panel_emb, use_cols)

    if USE_PERMUTATION:
        imp = permutation_importance_simple(panel_emb, use_cols)
        topk = imp.head(EMBEDDED_KEEP).index.tolist()
        embedded_imp = imp
    else:
        imp = train_lightgbm_embedded(panel_emb, use_cols)
        topk = imp.head(EMBEDDED_KEEP).index.tolist()
        embedded_imp = imp

    final_set = topk

    # Optional stability selection
    stability_counts = None
    if RUN_STABILITY:
        rng = np.random.default_rng(SEED)
        months_keys = sorted({int(y)*100 + int(m) for y, m in sel_months})
        counts = pd.Series(0, index=use_cols, dtype=int)
        for b in range(N_BOOTSTRAPS):
            # Sample months with replacement, keep order to respect blocks
            boot_months = rng.choice(months_keys, size=len(months_keys), replace=True)
            mask = (panel_emb["year"].astype(int)*100 + panel_emb["month"].astype(int)).isin(boot_months)
            boot = panel_emb.loc[mask].copy()
            if USE_PERMUTATION:
                b_imp = permutation_importance_simple(boot, use_cols)
                b_top = b_imp.head(STABILITY_TOPK).index
            else:
                b_imp = train_lightgbm_embedded(boot, use_cols)
                b_top = b_imp.head(STABILITY_TOPK).index
            counts[b_top] += 1
        freq = (counts / N_BOOTSTRAPS).sort_values(ascending=False).rename("freq")
        stable = freq[freq >= STABILITY_KEEP_AT_LEAST].index.tolist()
        # Intersect to be conservative
        final_set = [c for c in final_set if c in stable]
        stability_counts = freq

    # Save artifacts
    out_dir = Path("feature_selection_artifacts")
    out_dir.mkdir(exist_ok=True)
    ic_table.to_csv(out_dir / "rank_ic_table.csv")
    pd.Index(kept_after_corr).to_series().to_csv(out_dir / "kept_after_corr.csv", index=False)
    pd.Index(kept_after_ic).to_series().to_csv(out_dir / "kept_after_ic.csv", index=False)
    embedded_imp.to_csv(out_dir / ("importance_perm.csv" if USE_PERMUTATION else "importance_gain.csv"))
    pd.Index(final_set).to_series().to_csv(out_dir / "final_features.csv", index=False)
    if stability_counts is not None:
        stability_counts.to_csv(out_dir / "stability_freq.csv")

    # Write final feature list for downstream consumption
    column_file = Path("column_names.txt")
    column_file.write_text("\n".join(final_set) + ("\n" if final_set else ""))

    return {
        "after_correlation": kept_after_corr,
        "after_rank_ic": kept_after_ic,
        "final_features": final_set,
    }

def feature_select():
    run_pipeline(TrainWindow(start=(2005, 2), end=(2012, 12)))

if __name__ == "__main__":
    feature_select()