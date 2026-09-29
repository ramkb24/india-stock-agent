from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
PRICE_DIR = DATA_DIR / "prices"
INDICATOR_DIR = DATA_DIR / "indicators"
BENCHMARK_DIR = DATA_DIR / "benchmarks"

OUTPUT_DIR = DATA_DIR / "market"
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# BENCHMARK ANALYSIS
# =========================================================

def analyze_benchmark(
    name: str,
) -> dict:

    path = BENCHMARK_DIR / f"{name}.parquet"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing benchmark: {path}"
        )

    df = pd.read_parquet(path)

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    df["SMA_20"] = close.rolling(20).mean()
    df["SMA_50"] = close.rolling(50).mean()
    df["SMA_100"] = close.rolling(100).mean()
    df["SMA_200"] = close.rolling(200).mean()

    rsi = RSIIndicator(
        close=close,
        window=14,
    )

    df["RSI_14"] = rsi.rsi()

    atr = AverageTrueRange(
        high=high,
        low=low,
        close=close,
        window=14,
    )

    df["ATR_14"] = atr.average_true_range()

    latest = df.iloc[-1]

    close_price = float(latest["Close"])

    result = {
        "name": name,
        "date": df.index[-1],
        "close": close_price,

        "sma20": float(latest["SMA_20"]),
        "sma50": float(latest["SMA_50"]),
        "sma100": float(latest["SMA_100"]),
        "sma200": float(latest["SMA_200"]),

        "rsi": float(latest["RSI_14"]),
        "atr": float(latest["ATR_14"]),

        "above_sma20": close_price > latest["SMA_20"],
        "above_sma50": close_price > latest["SMA_50"],
        "above_sma100": close_price > latest["SMA_100"],
        "above_sma200": close_price > latest["SMA_200"],

        "sma20_above_sma50":
            latest["SMA_20"] > latest["SMA_50"],

        "sma50_above_sma200":
            latest["SMA_50"] > latest["SMA_200"],

        "return_1m":
            float(close.pct_change(21).iloc[-1] * 100),

        "return_3m":
            float(close.pct_change(63).iloc[-1] * 100),

        "return_6m":
            float(close.pct_change(126).iloc[-1] * 100),

        "return_1y":
            float(close.pct_change(252).iloc[-1] * 100),
    }

    return result


# =========================================================
# STOCK BREADTH
# =========================================================

def calculate_breadth() -> dict:

    files = sorted(
        INDICATOR_DIR.glob("*.parquet")
    )

    if not files:
        raise FileNotFoundError(
            "No technical indicator files found."
        )

    rows = []

    for file in files:

        try:

            df = pd.read_parquet(
                file,
                columns=[
                    "Close",
                    "SMA_20",
                    "SMA_50",
                    "SMA_200",
                    "TECHNICAL_SCORE",
                    "BREAKOUT_52W",
                    "BREAKDOWN_52W",
                ],
            )

            if df.empty:
                continue

            latest = df.iloc[-1]

            close = latest["Close"]

            rows.append(
                {
                    "symbol": file.stem,

                    "above_20dma":
                        close > latest["SMA_20"],

                    "above_50dma":
                        close > latest["SMA_50"],

                    "above_200dma":
                        close > latest["SMA_200"],

                    "technical_score":
                        latest["TECHNICAL_SCORE"],

                    "breakout":
                        bool(latest["BREAKOUT_52W"]),

                    "breakdown":
                        bool(latest["BREAKDOWN_52W"]),
                }
            )

        except Exception as exc:

            logger.warning(
                "Breadth error %s: %s",
                file.name,
                exc,
            )

    breadth = pd.DataFrame(rows)

    total = len(breadth)

    if total == 0:
        raise ValueError(
            "No stocks available for breadth calculation."
        )

    result = {
        "stocks": total,

        "above_20dma":
            int(breadth["above_20dma"].sum()),

        "above_50dma":
            int(breadth["above_50dma"].sum()),

        "above_200dma":
            int(breadth["above_200dma"].sum()),

        "pct_above_20dma":
            float(
                breadth["above_20dma"].mean() * 100
            ),

        "pct_above_50dma":
            float(
                breadth["above_50dma"].mean() * 100
            ),

        "pct_above_200dma":
            float(
                breadth["above_200dma"].mean() * 100
            ),

        "breakouts":
            int(breadth["breakout"].sum()),

        "breakdowns":
            int(breadth["breakdown"].sum()),

        "average_technical_score":
            float(
                breadth["technical_score"].mean()
            ),
    }

    return result


# =========================================================
# MARKET REGIME
# =========================================================

def determine_regime(
    nifty: dict,
    breadth: dict,
) -> tuple[str, float]:

    points = 0

    # ---------------------------------------------
    # Nifty trend
    # ---------------------------------------------

    if nifty["above_sma20"]:
        points += 1

    if nifty["above_sma50"]:
        points += 1

    if nifty["above_sma200"]:
        points += 2

    if nifty["sma20_above_sma50"]:
        points += 1

    if nifty["sma50_above_sma200"]:
        points += 2

    # ---------------------------------------------
    # Breadth
    # ---------------------------------------------

    if breadth["pct_above_50dma"] >= 60:
        points += 2
    elif breadth["pct_above_50dma"] >= 50:
        points += 1

    if breadth["pct_above_200dma"] >= 60:
        points += 2
    elif breadth["pct_above_200dma"] >= 50:
        points += 1

    # ---------------------------------------------
    # Momentum
    # ---------------------------------------------

    if nifty["return_3m"] > 0:
        points += 1

    if nifty["return_6m"] > 0:
        points += 1

    maximum = 13

    score = (
        points / maximum
    ) * 100

    # ---------------------------------------------
    # Regime
    # ---------------------------------------------

    if score >= 70:
        regime = "BULLISH"

    elif score >= 45:
        regime = "NEUTRAL"

    else:
        regime = "BEARISH"

    return regime, score


# =========================================================
# COMPLETE MARKET ANALYSIS
# =========================================================

def analyze_market() -> dict:

    print()
    print("=" * 70)
    print("INDIA STOCK AGENT — MARKET REGIME")
    print("=" * 70)
    print()

    benchmarks = {}

    for name in [
        "NIFTY50",
        "NIFTY500",
        "SENSEX",
    ]:

        print(
            f"Analyzing {name}..."
        )

        benchmarks[name] = analyze_benchmark(
            name
        )

    print(
        "Calculating market breadth..."
    )

    breadth = calculate_breadth()

    regime, regime_score = determine_regime(
        benchmarks["NIFTY50"],
        breadth,
    )

    result = {
        "regime": regime,
        "regime_score": regime_score,
        "benchmarks": benchmarks,
        "breadth": breadth,
    }

    return result


# =========================================================
# SAVE REPORT
# =========================================================

def save_market_report(
    result: dict,
) -> Path:

    output = OUTPUT_DIR / "market_regime.json"

    import json

    serializable = result.copy()

    for benchmark in serializable["benchmarks"].values():

        benchmark["date"] = str(
            benchmark["date"]
        )

    output.write_text(
        json.dumps(
            serializable,
            indent=2,
            default=str,
        )
    )

    return output


# =========================================================
# DISPLAY
# =========================================================

def print_report(
    result: dict,
) -> None:

    print()
    print("=" * 70)
    print("MARKET REGIME REPORT")
    print("=" * 70)

    print()
    print(
        f"REGIME       : {result['regime']}"
    )

    print(
        f"REGIME SCORE : "
        f"{result['regime_score']:.1f}/100"
    )

    print()

    print("BENCHMARKS")
    print("-" * 70)

    for name, data in result[
        "benchmarks"
    ].items():

        print()
        print(name)

        print(
            f"  Close       : {data['close']:.2f}"
        )

        print(
            f"  RSI         : {data['rsi']:.2f}"
        )

        print(
            f"  1M Return   : {data['return_1m']:.2f}%"
        )

        print(
            f"  3M Return   : {data['return_3m']:.2f}%"
        )

        print(
            f"  6M Return   : {data['return_6m']:.2f}%"
        )

        print(
            f"  1Y Return   : {data['return_1y']:.2f}%"
        )

        print(
            f"  >20 DMA     : {data['above_sma20']}"
        )

        print(
            f"  >50 DMA     : {data['above_sma50']}"
        )

        print(
            f"  >200 DMA    : {data['above_sma200']}"
        )

    print()
    print("MARKET BREADTH")
    print("-" * 70)

    breadth = result["breadth"]

    print(
        f"Stocks analyzed : {breadth['stocks']}"
    )

    print(
        f"> 20 DMA        : "
        f"{breadth['pct_above_20dma']:.1f}%"
    )

    print(
        f"> 50 DMA        : "
        f"{breadth['pct_above_50dma']:.1f}%"
    )

    print(
        f"> 200 DMA       : "
        f"{breadth['pct_above_200dma']:.1f}%"
    )

    print(
        f"52W breakouts   : "
        f"{breadth['breakouts']}"
    )

    print(
        f"52W breakdowns  : "
        f"{breadth['breakdowns']}"
    )

    print(
        f"Avg tech score  : "
        f"{breadth['average_technical_score']:.2f}"
    )

    print("=" * 70)


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    result = analyze_market()

    path = save_market_report(
        result
    )

    print_report(
        result
    )

    print()
    print(
        f"Saved report: {path}"
    )
