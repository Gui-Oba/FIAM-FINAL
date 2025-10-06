import pandas as pd
import numpy as np
from pathlib import Path

INPUT_ROOT = Path("../GUI_FINAL/ret_parquet")
OUTPUT_ROOT = Path("processed_data4")
ID_COLS = ["gvkey", "iid", "excntry"]
GROUP_CHOICES = [["year", "month"], ["year", "month"]]

YEAR_DTYPE = pd.Int32Dtype()
MONTH_DTYPE = pd.Int32Dtype()

LOW_PCT = 0.01
HIGH_PCT = 0.99
RET_CLIP_LOW = -1.0
RET_CLIP_HIGH = 3.0

RET_PREFIXES = ("stock_ret","ret_","rvol_","rskew_","ivol_","iskew_","coskew_","betadown_")
RATIO_HINTS = ("_me","_bev","_sale","_at","_be","_ev","_turnover","_var","_ratio")
CHANGE_PREFIXES = ("*_unused*",)  # placeholder so the tuple is non-empty
CHANGE_HINTS = ("_gr","_chg")


def is_return_col(c):
    return c == "stock_ret" or c.startswith(RET_PREFIXES)

def is_ratio_like(c):
    return any(h in c for h in RATIO_HINTS)

def is_change_col(c):
    return any(h in c for h in CHANGE_HINTS)

def cols_to_winsorize(all_num_cols):
    wins = []
    hard = []
    for c in all_num_cols:
        if c in ID_COLS:
            continue
        if c in ("year","month","y","m"):
            continue
        if is_return_col(c) or is_change_col(c) or is_ratio_like(c):
            wins.append(c)
        if is_return_col(c):
            hard.append(c)
    return wins, hard


def _coerce_month_keys(df: pd.DataFrame, month_dir: Path) -> None:
    """Ensure partition columns (`year`, `month`, `y`, `m`) are numeric and consistent."""

    def ensure_col(col_name: str, value: int, dtype) -> None:
        if col_name in df.columns:
            df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
            df[col_name] = df[col_name].fillna(value)
        else:
            df[col_name] = value
        df[col_name] = df[col_name].astype(dtype)

    year_val = int(month_dir.parent.name.split("=")[1])
    month_val = int(month_dir.name.split("=")[1])

    ensure_col("year", year_val, YEAR_DTYPE)
    ensure_col("month", month_val, MONTH_DTYPE)


def main():
    month_dirs = sorted(INPUT_ROOT.glob("y=*/m=*"))
    if not month_dirs:
        print(f"No parquet files found under {INPUT_ROOT}")
        return

    for month_dir in month_dirs:
        files = sorted(month_dir.glob("*.parquet"))
        if not files:
            continue

        frames = [pd.read_parquet(p) for p in files]
        df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
        df = df.copy()

        df = df.dropna(subset=ID_COLS)
        df.replace([np.inf,-np.inf], np.nan, inplace=True)

        group_cols = None
        for cols in GROUP_CHOICES:
            if all(col in df.columns for col in cols):
                group_cols = cols
                break
        if group_cols is None:
            group_cols = ["year","month"]

        _coerce_month_keys(df, month_dir)

        num_cols = [c for c in df.select_dtypes(include="number").columns if c not in ID_COLS and c not in group_cols]
        win_cols, hard_cols = cols_to_winsorize(num_cols)

        if win_cols:
            out_parts = []
            for _, g in df.groupby(group_cols):
                ql = g[win_cols].quantile(LOW_PCT)
                qh = g[win_cols].quantile(HIGH_PCT)
                h = g.copy()
                h[win_cols] = g[win_cols].clip(lower=ql, upper=qh, axis=1)
                if hard_cols:
                    for c in hard_cols:
                        if c in h.columns:
                            h[c] = h[c].clip(RET_CLIP_LOW, RET_CLIP_HIGH)
                out_parts.append(h)
            df = pd.concat(out_parts, ignore_index=True)

        df = df.dropna(subset=["stock_ret"]) if "stock_ret" in df.columns else df

        out_dir = OUTPUT_ROOT / month_dir.parent.name / month_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "part-0.parquet"

        # IDs: keep consistent types
        if "gvkey" in df:  df["gvkey"] = df["gvkey"].astype("int64")
        if "iid" in df:    df["iid"]   = df["iid"].astype("string")
        if "excntry" in df: df["excntry"] = df["excntry"].astype("string")

        # partitions
        if "year" in df: df["year"] = df["year"].astype("int32")
        if "month" in df: df["month"] = df["month"].astype("int32")

        # features: force float32 everywhere so NaN is allowed and schemas match
        num_cols_all = df.select_dtypes(include="number").columns.tolist()
        feature_cols = [c for c in num_cols_all if c not in ID_COLS and c not in ("year","month","y","m")]
        df[feature_cols] = df[feature_cols].astype("float32")

        df.to_parquet(out_path, index=False, engine="pyarrow")
        print(f"wrote {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
