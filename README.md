# Equity Return Modeling Pipeline

This repository hosts an end-to-end research pipeline for cross-sectional equity return prediction and portfolio evaluation. The workflow starts from a historical security-level returns CSV, engineers and filters features, trains predictive models with expanding windows, exports out-of-sample forecasts, and evaluates long–short portfolio performance.

The entry-point `main.py` executes every stage sequentially:

1. **Parquetize raw panel (`parquetization.py`)** – Uses DuckDB to convert `ret_sample.csv` into partitioned Parquet files (`ret_parquet/y=YYYY/m=MM`), adding `year`/`month` columns derived from the submission date.
2. **Preprocess features (`preprocess_parquets.py`)** – Cleans each monthly panel: drops malformed identifiers, handles inf/NaN, winsorizes and clips return-like columns, enforces canonical dtypes, and writes standardized Parquet shards under `processed_data4/`.
3. **Feature selection (`feature_select.py`)** – Loads the processed panel for a training window (default: Feb 2005 – Dec 2012) and performs:
   - Cross-sectional winsorization/z-scoring.
   - Low-variation and multicollinearity filters (mean absolute correlation threshold).
   - Predictive screening via rolling rank information coefficients to next-month returns.
   - Embedded model selection using XGBoost gain (or optional LightGBM permutation importance) with blocked time-series cross-validation.
   - Optional bootstrap stability selection.
   Artifacts (selected feature lists, IC tables, importances) land in `feature_selection_artifacts/`, while the final feature list is stored in `column_names.txt`.
4. **Model training (`model.py`)** – Builds a merged feature/return panel from `processed_data4/` and the original returns Parquet, performs cross-sectional ranking & scaling, then runs an expanding-window experiment:
   - PCA compression to retain ≥95% variance.
   - GPU-accelerated XGBoost regression (falls back to CPU if CuPy unavailable) tuned over a small grid, paired with a Ridge baseline.
   - Writes per-month out-of-sample predictions (including Ridge) into `oos_preds_pca/year=YYYY/month=MM/part-0.parquet`.
   - Prints summary statistics (R², PCA components, coverage) and saves detailed window summaries to stdout.
5. **Export predictions (`parquet_to_csv.py`)** – Consolidates the hierarchical Parquet predictions into a single CSV `predictions2.csv` using DuckDB.
6. **Portfolio analytics (`portfolio_optimization.py`)** – Consumes `predictions2.csv`, constructs equal-weighted decile portfolios from a smoothed signal, applies optional regime-aware blending (using `mkt_ind.csv` if present), and reports long–short Sharpe, CAPM alpha (Newey-West), drawdown, and turnover metrics.

Run `python main.py` to execute the full pipeline. Each module is also callable as a standalone script for ad-hoc experimentation.

---

## Project Layout

- `ret_sample.csv` – **User-supplied** raw panel with at least security identifiers (`gvkey`, `iid`, `excntry`), a `date` field in `YYYYMMDD` format, monthly `stock_ret`, and any candidate features.
- `parquetization.py` – DuckDB COPY into partitioned Parquet with `y`/`m` partitions.
- `processed_data4/` – Preprocessed, typed Parquet partitions consumed by downstream stages.
- `feature_selection_artifacts/` – CSV diagnostics for selection stages (`kept_after_corr.csv`, `rank_ic_table.csv`, etc.).
- `column_names.txt` – Line-separated list of vetted feature names ingested by `model.py`.
- `oos_preds_pca/` – Out-of-sample predictions grouped by `year`/`month`.
- `predictions2.csv` – Flat export of the Parquet predictions.
- `mkt_ind.csv` – Optional market-factor CSV (monthly) used for CAPM adjustments in the portfolio optimizer.

---

## Environment & Dependencies

Create a virtual environment (Python 3.10+ recommended) and install:

```bash
pip install duckdb pandas numpy pyarrow scipy scikit-learn xgboost statsmodels
```

Additional, optional packages:

- `cupy-cuda11x` (or the variant matching your CUDA toolkit) – accelerates XGBoost arrays if GPUs are available.
- `lightgbm` – required only when enabling permutation-based feature importance (`USE_PERMUTATION = True` in `feature_select.py`).

The preprocessing and modeling stages expect sufficient RAM to hold several months of panel data; GPU memory usage scales with the PCA output dimensionality.

---

## Data Requirements

The raw CSV should satisfy:

- `date` column formatted as `YYYYMMDD` (string or integer).
- Identifier columns: `gvkey`, `iid`, `excntry` (non-null after preprocessing).
- Target column: `stock_ret` (simple monthly return).
- Any number of numeric feature columns (ratios, changes, volatility measures, etc.). Columns containing `_me`, `_bev`, `_sale`, `_ret`, `_vol`, etc., are automatically considered for winsorization.

Preprocessing drops rows missing identifiers, replaces ±∞ with NaN, removes rows lacking `stock_ret`, and standardizes numeric dtypes to `float32`.

---

## Usage

### Run the full pipeline

```bash
python main.py
```

The run produces (in order):

1. `ret_parquet/` (partitioned raw panel)
2. `processed_data4/` (clean inputs) + console logs per month
3. `feature_selection_artifacts/` and `column_names.txt`
4. `oos_preds_pca/` Parquet predictions with summary metrics
5. `predictions2.csv`
6. Portfolio analytics printed to stdout

Stage outputs are idempotent—re-running overwrites files in place.

### Execute individual modules

Invoke any script directly to rerun only that stage:

```bash
python parquetization.py
python preprocess_parquets.py
python feature_select.py
python model.py
python parquet_to_csv.py
python portfolio_optimization.py
```

Ensure prerequisite artifacts from earlier stages are present (e.g., `processed_data4/` before `feature_select.py`).

---

## Configuration Highlights

- **`parquetization.py`** – Adjust the DuckDB SQL if your raw file name or date format differs.
- **`preprocess_parquets.py`** – Tweak winsorization thresholds (`LOW_PCT`, `HIGH_PCT`), hard return clipping (`RET_CLIP_LOW/HIGH`), or extend identifier lists if new keys are introduced.
- **`feature_select.py`** – Change `TrainWindow(start, end)` to modify the training span; update thresholds (`ABS_R_THRESHOLD`, `MAX_AFTER_IC`, `EMBEDDED_KEEP`) as needed. Set `USE_PERMUTATION = True` to switch to LightGBM-based permutation importance.
- **`model.py`** – Edit PCA variance target (`PCA_VARIANCE`), parameter grid (`PARAM_GRID`), ridge alphas, or expanding-window horizons (`TRAIN_YEARS`, `VAL_YEARS`, `TEST_YEARS`). GPU use is automatic when CuPy detects a CUDA device.
- **`portfolio_optimization.py`** – Swap `MODEL_COL` to evaluate a different signal (e.g., `predicted_stock_ret`), adjust smoothing half-life (`FIXED_HALF_LIFE`), or provide a market factors file via `MKT_FILE`.

---

## Troubleshooting

- **Missing directories/files** – Ensure each prior stage ran successfully; most scripts raise a descriptive `FileNotFoundError`.
- **DuckDB errors** – Confirm `ret_sample.csv` exists and has a `date` column convertible via `STRPTIME(..., '%Y%m%d')`.
- **Feature selection failures** – Review `feature_selection_artifacts/kept_after_corr.csv` to diagnose empty selections. Expand the training window or relax thresholds if too few features survive.
- **GPU issues** – If CUDA is unavailable, XGBoost automatically copies NumPy arrays to the GPU; to force CPU, remove `device="cuda"` in `model.py`.
- **Portfolio CAPM step** – Provide `mkt_ind.csv` with columns `year`, `month`, and `mkt_rf` (or `ret`/`rf`, or `mktrf`). Otherwise the CAPM section is skipped.

---

## Next Steps

- Augment logging with structured metrics (e.g., CSV summaries from `model.py`).
- Add unit or integration tests covering preprocessing edge cases.
- Package the pipeline as a CLI or workflow tool for reproducible research runs.
