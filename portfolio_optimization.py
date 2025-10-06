# === Adaptive Single→Multi Horizon (Causal, No In-Sample Tuning) ===
# - Causal regime detection (uses info up to end of t-1 for decisions in month t)
# - No filtering on realized returns before ranking (no look-ahead)
# - Fixed half-life (constant alpha) to avoid in-sample tuning
#
# INPUT:
#   columns (any case): gvkey, iid, year, month,
#                       predicted_stock_ret (signal), actual_stock_ret (realized simple monthly return)
#
# OPTIONAL:
#   MKT_FILE: CSV with year, month, and either mkt_rf or (ret, rf) or mktrf (for CAPM alpha with NW-HAC)
#   INDEX_FILE:
#       - Daily: ['date','close'] (or 'spx' instead of 'close')
#       - Monthly: ['year','month','close'] (or 'spx' instead of 'close')
#
# OUTPUT: same labels as your old code (Sharpe, CAPM summary, Alpha, etc.)

import os
import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthBegin
import statsmodels.formula.api as sm

def portfolio():
    # ---------- CONFIG ----------
    PRED_FILE  = "predictions2.csv"
    MODEL_COL  = "ridge_stock_ret"     # 1m signal
    RET_COL    = "actual_stock_ret"        # realized simple monthly return
    GVKEY_COL  = "gvkey"
    IID_COL    = "iid"
    YEAR_COL   = "year"
    MONTH_COL  = "month"

    # Market excess return file (optional; for CAPM alpha)
    MKT_FILE   = "mkt_ind.csv"

    # Fixed half-life (months) to avoid in-sample tuning
    FIXED_HALF_LIFE = 6.0
    ALPHA = 1.0 - 2.0**(-1.0 / FIXED_HALF_LIFE)

    # Which signal to build deciles from: "s_star" (default) or "pred"
    DECILE_SOURCE = "s_star"

    # ---------- LOAD ----------
    df = pd.read_csv(PRED_FILE)
    df.columns = [c.strip().lower() for c in df.columns]

    need = {GVKEY_COL, IID_COL, YEAR_COL, MONTH_COL, MODEL_COL, RET_COL}
    missing = need - set(df.columns)
    assert not missing, f"Missing columns: {missing}"

    # Build id and date
    df["id"] = df[GVKEY_COL].astype(str) + "-" + df[IID_COL].astype(str)
    df["date"] = pd.to_datetime(dict(year=df[YEAR_COL].astype(int),
                                    month=df[MONTH_COL].astype(int), day=1))

    # --- Keep realized returns; don't pre-filter by them (avoid look-ahead) ---
    df = df[["id","date",YEAR_COL,MONTH_COL,MODEL_COL,RET_COL]].copy()
    # Only require the signal to exist at formation time:
    df = df[df[MODEL_COL].notna()].sort_values(["date","id"])

    # ---------- Cross-sectional ranks & z-scores (by month) ----------
    def zscore(x):
        mu, sd = x.mean(), x.std(ddof=0)
        return (x - mu) / (sd if sd > 0 else 1.0)

    df["rank_s"] = df.groupby("date")[MODEL_COL].rank(method="first")  # 1..N
    df["z_s"]    = df.groupby("date")[MODEL_COL].transform(zscore)

    # ---------- Exponential smoothing with FIXED alpha (causal) ----------
    # adjust=False ensures recursion uses only past/current values
    df = df.sort_values(["id","date"])
    df["z_s_sm"] = df.groupby("id", group_keys=False)["z_s"].apply(
        lambda s: s.ewm(alpha=ALPHA, adjust=False).mean()
    )

    # ---------- Causal regime detection ----------
    def build_regime_causal(mkt_file):
        
        # Fallback: market realized vol tercile, all computed on history up to t-1
        if mkt_file and os.path.exists(mkt_file):
            mkt = pd.read_csv(mkt_file)
            mkt.columns = [c.strip().lower() for c in df.columns] if False else [c.strip().lower() for c in mkt.columns]
            if "mkt_rf" not in mkt.columns:
                if {"ret","rf"}.issubset(mkt.columns):
                    mkt["mkt_rf"] = mkt["ret"] - mkt["rf"]
                elif "mktrf" in mkt.columns:
                    mkt["mkt_rf"] = mkt["mktrf"]
                else:
                    raise ValueError("Need `mkt_rf` or (`ret` and `rf`) or `mktrf` in factors file.")
            mkt["date"] = pd.to_datetime(dict(year=mkt["year"].astype(int),
                                            month=mkt["month"].astype(int), day=1))
            mkt = mkt.sort_values("date")

            # vol as of t-1
            rv_12m_hist = mkt["mkt_rf"].rolling(12, min_periods=6).std(ddof=0).shift(1)

            # expanding percentile computed on strictly past values (up to i-1)
            def expanding_pctile_past(x, q=0.667):
                out = []
                for i in range(len(x)):
                    if i < 7:  # need ~6 past obs after the shift
                        out.append(np.nan)
                    else:
                        out.append(np.nanpercentile(x[:i], q*100))
                return pd.Series(out, index=x.index)

            mkt["rv_12m_p67_prev"] = expanding_pctile_past(rv_12m_hist)
            mkt["defensive"] = (rv_12m_hist >= mkt["rv_12m_p67_prev"]).astype(int)
            return mkt[["date","defensive"]]

        return None

    regime = build_regime_causal(MKT_FILE)

    if regime is not None:
        # Merge onto scores (date must be first-of-month for t), fill missing as risk-on
        reg_months = int(regime["defensive"].notna().sum())
        reg_def = int(regime["defensive"].fillna(0).sum())
        pct = reg_def / reg_months if reg_months else float("nan")
        print(f"Defensive months: {reg_def}/{reg_months} ({pct:.1%} of sample)")

    # ---------- Blend scores (causal, regime-aware) ----------
    df_scores = df[["id","date","z_s","z_s_sm"]].copy()
    if regime is not None:
        df_scores = df_scores.merge(regime, on="date", how="left")
        df_scores["defensive"] = df_scores["defensive"].fillna(0)
        df_scores["s_star"] = np.where(
            df_scores["defensive"] > 0,
            0.4*df_scores["z_s"] + 0.6*df_scores["z_s_sm"],   # defensive → lean slower
            0.7*df_scores["z_s"] + 0.3*df_scores["z_s_sm"],   # risk-on → lean faster
        )
    else:
        df_scores["s_star"] = 0.6*df_scores["z_s_sm"] + 0.4*df_scores["z_s"]

    # ---------- Evaluation (same metrics/labels/order) ----------
    base = df.merge(df_scores[["id","date","s_star"]], on=["id","date"], how="left")
    signal_col = "s_star" if DECILE_SOURCE == "s_star" else MODEL_COL

    # Deciles within year-month (equal-weight). No peek at future.
    grp = base.groupby([YEAR_COL, MONTH_COL], group_keys=False)[signal_col]
    base["rank"] = np.floor(
        grp.transform(lambda s: s.rank(method="first")) * 10 / (grp.transform("size") + 1)
    ).astype(int).clip(0, 9)  # 0..9

    # EQUAL-WEIGHT RETURNS PER DECILE (NaNs in returns are naturally ignored)
    monthly_port = (
        base.groupby([YEAR_COL, MONTH_COL, "rank"], as_index=False)[RET_COL].mean()
            .pivot(index=[YEAR_COL, MONTH_COL], columns="rank", values=RET_COL)
            .rename(columns={i: f"port_{i+1}" for i in range(10)})  # 0->port_1 ... 9->port_10
            .sort_index()
    )

    # Require both legs to exist that month (still causal)
    monthly_port = monthly_port.dropna(subset=["port_1","port_10"], how="any")
    monthly_port["port_11"] = monthly_port["port_10"] - monthly_port["port_1"]

    # ---------- Sharpe (annualized, monthly) ----------
    sr = monthly_port["port_11"].mean() / monthly_port["port_11"].std(ddof=1) * np.sqrt(12)
    print(f"Sharpe Ratio (LS, annualized): {sr:.4f}")

    # ---------- CAPM alpha with Newey-West (optional) ----------
    if MKT_FILE and os.path.exists(MKT_FILE):
        mkt = pd.read_csv(MKT_FILE)
        mkt.columns = [c.strip().lower() for c in mkt.columns]

        if "mkt_rf" not in mkt.columns:
            if {"ret", "rf"}.issubset(mkt.columns):
                mkt["mkt_rf"] = mkt["ret"] - mkt["rf"]
            elif "mktrf" in mkt.columns:
                mkt["mkt_rf"] = mkt["mktrf"]
            else:
                raise ValueError("Need `mkt_rf` or (`ret` and `rf`) or `mktrf` in factors file.")

        df_capm = monthly_port.reset_index().rename(columns={YEAR_COL: "year", MONTH_COL: "month"})
        df_capm["year"]  = df_capm["year"].astype(int)
        df_capm["month"] = df_capm["month"].astype(int)
        mkt["year"]      = mkt["year"].astype(int)
        mkt["month"]     = mkt["month"].astype(int)

        df_capm = df_capm.merge(mkt[["year", "month", "mkt_rf"]], on=["year", "month"], how="inner")
        nw_ols = sm.ols("port_11 ~ mkt_rf", data=df_capm).fit(
            cov_type="HAC", cov_kwds={"maxlags": 3}, use_t=True
        )
        print(nw_ols.summary())

        alpha_capm   = nw_ols.params["Intercept"]
        t_alpha      = nw_ols.tvalues["Intercept"]
        info_ratio   = alpha_capm / np.sqrt(nw_ols.mse_resid) * np.sqrt(12)
        print(f"CAPM Alpha (monthly): {alpha_capm:.6f}")
        print(f"t-stat(Alpha): {t_alpha:.3f}")
        print(f"Information Ratio (annualized): {info_ratio:.4f}")
    else:
        print("CAPM step skipped (set MKT_FILE to your factors CSV).")

    # ---------- Risk stats ----------
    max_1m_loss = monthly_port["port_11"].min()
    print(f"Max 1-Month Loss (LS): {max_1m_loss:.4f}")

    mp = monthly_port.copy()
    mp["log_ls"] = np.log1p(mp["port_11"])
    mp["cumlog_ls"] = mp["log_ls"].cumsum()
    drawdown = (mp["cumlog_ls"].cummax() - mp["cumlog_ls"]).max()
    print(f"Maximum Drawdown (log space): {drawdown:.4f}")

    # ---------- Turnover (D10 & D1) ----------
    def turnover_count(df_subset: pd.DataFrame) -> float:
        if df_subset.empty:
            return float("nan")
        tmp = df_subset[["id","date"]].drop_duplicates().sort_values(["date","id"])
        start = tmp.groupby("date")["id"].apply(set).rename("members")
        tmp2 = tmp.copy()
        tmp2["date"] = (tmp2["date"] - MonthBegin(1)).dt.to_period("M").dt.to_timestamp()
        end = tmp2.groupby("date")["id"].apply(set).rename("members_next")
        counts = pd.concat([start, end], axis=1).dropna()
        if counts.empty:
            return float("nan")
        def one_turn(row):
            s, e = row["members"], row["members_next"]
            return (len(s - e) / len(s)) if len(s) else np.nan
        return float(counts.apply(one_turn, axis=1).mean())

    sel = base[["id","date","rank"]].copy()
    long_df  = sel[sel["rank"] == 9]
    short_df = sel[sel["rank"] == 0]
    print(f"Long (D10) avg monthly turnover:  {turnover_count(long_df):.4f}")
    print(f"Short (D1) avg monthly turnover: {turnover_count(short_df):.4f}")

if __name__ == "__main__":
    portfolio()