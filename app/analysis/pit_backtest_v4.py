"""
PIT backtest v4 — three-factor composite with buffered selection.

Changes vs v3:
  1. Composite = 0.45*technical + 0.35*relative_strength + 0.20*fundamental
  2. Each factor is percentile-ranked cross-sectionally before blending
  3. Selection uses ranks [10:10+N] instead of nlargest(N)
  4. Prints decile curve, rank-tier table, yearly breakdown
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path.home() / "india-stock-agent"
sys.path.insert(0, str(ROOT))

from app.analysis.technical_scoring import calculate_technical_components

PIT_UNIVERSE = ROOT / "data" / "backtests" / "pit_universe" / "universe_pit_v2.csv"
PRICE_DIR = ROOT / "data" / "prices_v2"
SCORES_CSV = ROOT / "data" / "market" / "stock_scores.csv"
OUT_DIR = ROOT / "data" / "backtests" / "pit_universe"

START_DATE = pd.Timestamp("2022-03-22")
END_DATE = pd.Timestamp("2026-09-25")
REBALANCE_DAYS = 21
PORTFOLIO_SIZES = [10, 20, 50]
BUFFER = 10   # skip the top N ranked stocks
WEIGHTS = {
    "TECH": 0.45,
    "RS":   0.35,
    "FUND": 0.20,
}

# ------------------------------------------------------------
# Load snapshot RS and FUND (constant per symbol)
# ------------------------------------------------------------
scores = pd.read_csv(SCORES_CSV)
scores["symbol_clean"] = (
    scores["symbol"].astype(str).str.replace("_NS", "", regex=False)
)

def factor_map(col):
    s = pd.to_numeric(scores[col], errors="coerce")
    return dict(zip(scores["symbol_clean"], s))

RS_MAP = factor_map("relative_strength_score")
FUND_MAP = factor_map("fundamental_score_final")

# ------------------------------------------------------------
# Load prices, compute technical score history
# ------------------------------------------------------------
pit = pd.read_csv(PIT_UNIVERSE)
symbols = sorted(set(pit["yahoo_symbol"].astype(str).str.strip()))

raw = {}
closes = {}
rs_static = {}
fund_static = {}

for symbol in symbols:
    safe = symbol.replace(".", "_").replace("/", "_")
    path = PRICE_DIR / f"{safe}.parquet"
    if not path.exists():
        continue
    df = pd.read_parquet(path)
    if df.empty or len(df) < 300:
        continue

    idx = pd.to_datetime(df.index, errors="coerce")
    df = df.loc[~idx.isna()].copy()
    idx = pd.DatetimeIndex(idx[~idx.isna()])
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df.index = idx
    df = df.sort_index()

    comp = calculate_technical_components(df)
    raw[symbol] = comp["SCORE_RAW"].loc[START_DATE:END_DATE]
    closes[symbol] = df["Close"].loc[START_DATE:END_DATE]

    clean = symbol.replace(".NS", "")
    rs_static[symbol] = RS_MAP.get(clean, np.nan)
    fund_static[symbol] = FUND_MAP.get(clean, np.nan)

print(f"Loaded {len(closes)} symbols")

# ------------------------------------------------------------
# Build factor matrices on the same date index
# ------------------------------------------------------------
tech_raw = pd.DataFrame(raw)
close_matrix = pd.DataFrame(closes)
dates = tech_raw.index

# Percentile-rank each factor per date
tech_rank = tech_raw.rank(axis=1, pct=True) * 100.0

# Static factors broadcast across all dates
rs_row = pd.Series({s: rs_static.get(s, np.nan) for s in tech_raw.columns})
fund_row = pd.Series({s: fund_static.get(s, np.nan) for s in tech_raw.columns})

rs_rank = pd.DataFrame([rs_row] * len(dates), index=dates).rank(axis=1, pct=True) * 100.0
fund_rank = pd.DataFrame([fund_row] * len(dates), index=dates).rank(axis=1, pct=True) * 100.0

# Composite
composite = (
    tech_rank * WEIGHTS["TECH"]
    + rs_rank * WEIGHTS["RS"]
    + fund_rank * WEIGHTS["FUND"]
)

# Re-rank composite to 0-100
composite = composite.rank(axis=1, pct=True) * 100.0

print(f"composite shape: {composite.shape}")

# ------------------------------------------------------------
# Diagnostics
# ------------------------------------------------------------
rebalance_dates = dates[::REBALANCE_DAYS]

print()
print("=" * 70)
print("SATURATION CHECK")
print("=" * 70)
tied = []
for rd in rebalance_dates[:-1]:
    row = composite.loc[rd].dropna()
    if len(row) < 20:
        continue
    tied.append((row == row.max()).sum())
print(f"mean tied at max : {np.mean(tied):.2f}")
print(f"max  tied at max : {np.max(tied)}")

# ------------------------------------------------------------
# Simulation
# ------------------------------------------------------------
def forward_return(signal_date, next_signal_date, selected):
    future = dates[dates > signal_date]
    if len(future) == 0:
        return None
    entry = future[0]
    exit_c = dates[dates >= next_signal_date]
    if len(exit_c) == 0:
        return None
    exit_d = exit_c[0]

    rets = []
    for sym in selected:
        if sym not in close_matrix.columns:
            continue
        c = close_matrix[sym]
        if entry not in c.index or exit_d not in c.index:
            continue
        ep = c.loc[entry]
        xp = c.loc[exit_d]
        if pd.isna(ep) or pd.isna(xp) or ep <= 0:
            continue
        rets.append(xp / ep - 1)
    if not rets:
        return None
    return float(np.mean(rets))

def simulate(portfolio_size, reverse=False, buffered=True):
    rows = []
    for i in range(len(rebalance_dates) - 1):
        sd = rebalance_dates[i]
        nsd = rebalance_dates[i + 1]

        row = composite.loc[sd].dropna()
        if len(row) < (BUFFER + portfolio_size):
            continue

        ranked = row.sort_values(ascending=False).index.tolist()

        if reverse:
            selected = ranked[-portfolio_size:]  # bottom N
        else:
            if buffered:
                selected = ranked[BUFFER:BUFFER + portfolio_size]
            else:
                selected = ranked[:portfolio_size]

        r = forward_return(sd, nsd, selected)
        if r is None:
            continue

        rows.append({
            "signal_date": sd,
            "return": r - 0.0020,
        })
    return pd.DataFrame(rows)

def metrics(returns):
    returns = returns.dropna()
    if len(returns) == 0:
        return {}
    equity = (1 + returns).cumprod()
    total = equity.iloc[-1] - 1
    years = len(returns) * REBALANCE_DAYS / 252
    cagr = equity.iloc[-1] ** (1 / years) - 1
    periods_per_year = 252.0 / REBALANCE_DAYS
    vol = returns.std() * (periods_per_year ** 0.5)
    sharpe = (returns.mean() / returns.std() * (periods_per_year ** 0.5)
              if returns.std() > 0 else 0)
    dd = (equity / equity.cummax() - 1).min()
    return {
        "total_return_pct": total * 100,
        "cagr_pct": cagr * 100,
        "annualized_volatility_pct": vol * 100,
        "sharpe": sharpe,
        "max_drawdown_pct": dd * 100,
        "win_rate_pct": (returns > 0).mean() * 100,
        "periods": len(returns),
    }

print()
print("=" * 90)
print("RESULTS")
print("=" * 90)

results = []

# Unbuffered top / bottom
for size in PORTFOLIO_SIZES:
    r = simulate(size, reverse=False, buffered=False)
    m = metrics(r["return"])
    m["strategy"] = f"UNBUFFERED_TOP_{size}"
    results.append(m)

    r = simulate(size, reverse=True, buffered=False)
    m = metrics(r["return"])
    m["strategy"] = f"BOTTOM_{size}"
    results.append(m)

# Buffered top
for size in PORTFOLIO_SIZES:
    r = simulate(size, reverse=False, buffered=True)
    m = metrics(r["return"])
    m["strategy"] = f"BUFFERED_TOP_{size}"
    results.append(m)

out = pd.DataFrame(results)[[
    "strategy", "total_return_pct", "cagr_pct",
    "annualized_volatility_pct", "sharpe",
    "max_drawdown_pct", "win_rate_pct", "periods",
]]
print(out.to_string(index=False))

# ------------------------------------------------------------
# Decile curve
# ------------------------------------------------------------
print()
print("=" * 80)
print("DECILE CURVE (composite)")
print("=" * 80)

decile_rets = {d: [] for d in range(1, 11)}
for i in range(len(rebalance_dates) - 1):
    sd = rebalance_dates[i]
    nsd = rebalance_dates[i + 1]

    row = composite.loc[sd].dropna()
    if len(row) < 100:
        continue

    deciles = pd.qcut(row, 10, labels=False) + 1

    for d in range(1, 11):
        syms = deciles[deciles == d].index.tolist()
        r = forward_return(sd, nsd, syms)
        if r is not None:
            decile_rets[d].append(r)

print(f"{'Decile':>8s}  {'n':>4s}  {'mean':>9s}  {'hit%':>6s}  {'cumulative':>11s}")
for d in range(1, 11):
    arr = np.array(decile_rets[d])
    if len(arr) == 0:
        continue
    label = f"D{d}"
    if d == 1:
        label += " (LOW)"
    if d == 10:
        label += " (HIGH)"
    print(f"{label:>13s}  {len(arr):4d}  "
          f"{arr.mean()*100:+8.3f}%  "
          f"{(arr>0).mean()*100:5.1f}%  "
          f"{(np.prod(1+arr)-1)*100:+10.2f}%")

# ------------------------------------------------------------
# Yearly breakdown of BUFFERED_TOP_10 vs BOTTOM_10
# ------------------------------------------------------------
print()
print("=" * 80)
print("YEARLY: BUFFERED_TOP_10 vs BOTTOM_10")
print("=" * 80)

r_top = simulate(10, reverse=False, buffered=True)
r_bot = simulate(10, reverse=True, buffered=False)

r_top["year"] = pd.to_datetime(r_top["signal_date"]).dt.year
r_bot["year"] = pd.to_datetime(r_bot["signal_date"]).dt.year

top_y = r_top.groupby("year")["return"].apply(lambda x: (1 + x).prod() - 1) * 100
bot_y = r_bot.groupby("year")["return"].apply(lambda x: (1 + x).prod() - 1) * 100

combined = pd.DataFrame({"TOP_BUFFERED": top_y, "BOTTOM": bot_y})
combined["SPREAD"] = combined["TOP_BUFFERED"] - combined["BOTTOM"]
print(combined.to_string(float_format=lambda x: f"{x:+.2f}%"))

out.to_csv(OUT_DIR / "pit_v4_backtest_summary.csv", index=False)
print()
print(f"Saved: {OUT_DIR / 'pit_v4_backtest_summary.csv'}")
