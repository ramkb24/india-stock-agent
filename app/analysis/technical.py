from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ta.momentum import (
    ROCIndicator,
    RSIIndicator,
    StochasticOscillator,
)
from ta.trend import (
    ADXIndicator,
    MACD,
)
from ta.volatility import (
    AverageTrueRange,
    BollingerBands,
)
from ta.volume import VolumeWeightedAveragePrice

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PRICE_DIR = PROJECT_ROOT / "data" / "prices"
INDICATOR_DIR = PROJECT_ROOT / "data" / "indicators"

INDICATOR_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# LOAD PRICE DATA
# =========================================================

def load_price_file(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)

    if df.empty:
        return df

    df = df.copy()

    df.index = pd.to_datetime(
        df.index,
        errors="coerce",
    )

    df = df[~df.index.isna()]

    df = df.sort_index()

    return df


# =========================================================
# TECHNICAL INDICATORS
# =========================================================

def calculate_indicators(
    df: pd.DataFrame,
) -> pd.DataFrame:

    if df.empty:
        return df

    df = df.copy()

    required = {
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing price columns: {sorted(missing)}"
        )

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # -----------------------------------------------------
    # Moving averages
    # -----------------------------------------------------

    df["SMA_20"] = close.rolling(20).mean()
    df["SMA_50"] = close.rolling(50).mean()
    df["SMA_100"] = close.rolling(100).mean()
    df["SMA_200"] = close.rolling(200).mean()

    df["EMA_20"] = close.ewm(
        span=20,
        adjust=False,
    ).mean()

    df["EMA_50"] = close.ewm(
        span=50,
        adjust=False,
    ).mean()

    # -----------------------------------------------------
    # RSI
    # -----------------------------------------------------

    rsi = RSIIndicator(
        close=close,
        window=14,
    )

    df["RSI_14"] = rsi.rsi()

    # -----------------------------------------------------
    # MACD
    # -----------------------------------------------------

    macd = MACD(
        close=close,
        window_slow=26,
        window_fast=12,
        window_sign=9,
    )

    df["MACD"] = macd.macd()
    df["MACD_SIGNAL"] = macd.macd_signal()
    df["MACD_HIST"] = macd.macd_diff()

    # -----------------------------------------------------
    # ADX
    # -----------------------------------------------------

    adx = ADXIndicator(
        high=high,
        low=low,
        close=close,
        window=14,
    )

    df["ADX_14"] = adx.adx()
    df["DI_PLUS"] = adx.adx_pos()
    df["DI_MINUS"] = adx.adx_neg()

    # -----------------------------------------------------
    # ATR
    # -----------------------------------------------------

    atr = AverageTrueRange(
        high=high,
        low=low,
        close=close,
        window=14,
    )

    df["ATR_14"] = atr.average_true_range()

    df["ATR_PERCENT"] = (
        df["ATR_14"] / close * 100
    )

    # -----------------------------------------------------
    # Bollinger Bands
    # -----------------------------------------------------

    bb = BollingerBands(
        close=close,
        window=20,
        window_dev=2,
    )

    df["BB_MIDDLE"] = bb.bollinger_mavg()
    df["BB_HIGH"] = bb.bollinger_hband()
    df["BB_LOW"] = bb.bollinger_lband()
    df["BB_WIDTH"] = bb.bollinger_wband()
    df["BB_PERCENT"] = bb.bollinger_pband()

    # -----------------------------------------------------
    # Stochastic
    # -----------------------------------------------------

    stochastic = StochasticOscillator(
        high=high,
        low=low,
        close=close,
        window=14,
        smooth_window=3,
    )

    df["STOCH_K"] = stochastic.stoch()
    df["STOCH_D"] = stochastic.stoch_signal()

    # -----------------------------------------------------
    # Rate of Change
    # -----------------------------------------------------

    roc = ROCIndicator(
        close=close,
        window=20,
    )

    df["ROC_20"] = roc.roc()

    # -----------------------------------------------------
    # Volume
    # -----------------------------------------------------

    df["VOLUME_SMA_20"] = volume.rolling(20).mean()
    df["VOLUME_SMA_50"] = volume.rolling(50).mean()

    df["RELATIVE_VOLUME"] = (
        volume / df["VOLUME_SMA_20"]
    )

    # -----------------------------------------------------
    # VWAP
    # -----------------------------------------------------

    vwap = VolumeWeightedAveragePrice(
        high=high,
        low=low,
        close=close,
        volume=volume,
        window=20,
    )

    df["VWAP_20"] = vwap.volume_weighted_average_price()

    # -----------------------------------------------------
    # Returns
    # -----------------------------------------------------

    df["RETURN_1D"] = close.pct_change(1) * 100
    df["RETURN_5D"] = close.pct_change(5) * 100
    df["RETURN_20D"] = close.pct_change(20) * 100
    df["RETURN_60D"] = close.pct_change(60) * 100
    df["RETURN_120D"] = close.pct_change(120) * 100
    df["RETURN_252D"] = close.pct_change(252) * 100

    # -----------------------------------------------------
    # 52-week structure
    # -----------------------------------------------------

    df["HIGH_52W"] = (
        high.rolling(252).max()
    )

    df["LOW_52W"] = (
        low.rolling(252).min()
    )

    df["DIST_FROM_52W_HIGH"] = (
        (close / df["HIGH_52W"]) - 1
    ) * 100

    df["DIST_FROM_52W_LOW"] = (
        (close / df["LOW_52W"]) - 1
    ) * 100

    # -----------------------------------------------------
    # Price vs moving averages
    # -----------------------------------------------------

    df["ABOVE_SMA_20"] = (
        close > df["SMA_20"]
    )

    df["ABOVE_SMA_50"] = (
        close > df["SMA_50"]
    )

    df["ABOVE_SMA_100"] = (
        close > df["SMA_100"]
    )

    df["ABOVE_SMA_200"] = (
        close > df["SMA_200"]
    )

    # -----------------------------------------------------
    # Trend structure
    # -----------------------------------------------------

    df["BULLISH_ALIGNMENT"] = (
        (df["SMA_20"] > df["SMA_50"])
        & (df["SMA_50"] > df["SMA_100"])
        & (df["SMA_100"] > df["SMA_200"])
    )

    df["BEARISH_ALIGNMENT"] = (
        (df["SMA_20"] < df["SMA_50"])
        & (df["SMA_50"] < df["SMA_100"])
        & (df["SMA_100"] < df["SMA_200"])
    )

    # -----------------------------------------------------
    # MACD state
    # -----------------------------------------------------

    df["MACD_BULLISH"] = (
        df["MACD"] > df["MACD_SIGNAL"]
    )

    # -----------------------------------------------------
    # ADX state
    # -----------------------------------------------------

    df["STRONG_TREND"] = (
        df["ADX_14"] >= 25
    )

    df["ADX_BULLISH"] = (
        (df["ADX_14"] >= 20)
        & (df["DI_PLUS"] > df["DI_MINUS"])
    )

    # -----------------------------------------------------
    # Breakouts
    #
    # Compare today's price with the previous 252
    # observations, deliberately excluding today.
    # This prevents look-ahead in the breakout calculation.
    # -----------------------------------------------------

    previous_52w_high = (
        high.shift(1)
        .rolling(252)
        .max()
    )

    previous_52w_low = (
        low.shift(1)
        .rolling(252)
        .min()
    )

    df["BREAKOUT_52W"] = (
        close > previous_52w_high
    )

    df["BREAKDOWN_52W"] = (
        close < previous_52w_low
    )

    # -----------------------------------------------------
    # Volume confirmation
    # -----------------------------------------------------

    df["HIGH_VOLUME"] = (
        df["RELATIVE_VOLUME"] >= 1.5
    )

    df["VERY_HIGH_VOLUME"] = (
        df["RELATIVE_VOLUME"] >= 2.0
    )

    # -----------------------------------------------------
    # Volatility regime
    # -----------------------------------------------------

    df["VOLATILITY_20D"] = (
        close.pct_change()
        .rolling(20)
        .std()
        * np.sqrt(252)
        * 100
    )

    # -----------------------------------------------------
    # Trend label
    # -----------------------------------------------------

    conditions = [
        df["BULLISH_ALIGNMENT"],
        df["BEARISH_ALIGNMENT"],
    ]

    choices = [
        "BULLISH",
        "BEARISH",
    ]

    df["TREND"] = np.select(
        conditions,
        choices,
        default="NEUTRAL",
    )

    return df


# =========================================================
# TECHNICAL SCORE
# =========================================================

def calculate_score(df: pd.DataFrame) -> pd.DataFrame:

    if df.empty:
        return df

    df = df.copy()

    score = pd.Series(
        0.0,
        index=df.index,
    )

    # -----------------------------------------------------
    # Trend — 25 points
    # -----------------------------------------------------

    trend_score = pd.Series(
        0.0,
        index=df.index,
    )

    trend_score += (
        df["ABOVE_SMA_20"].astype(float) * 5
    )

    trend_score += (
        df["ABOVE_SMA_50"].astype(float) * 5
    )

    trend_score += (
        df["ABOVE_SMA_100"].astype(float) * 5
    )

    trend_score += (
        df["ABOVE_SMA_200"].astype(float) * 5
    )

    trend_score += (
        df["BULLISH_ALIGNMENT"].astype(float) * 5
    )

    # -----------------------------------------------------
    # Momentum — 20 points
    # -----------------------------------------------------

    momentum_score = pd.Series(
        0.0,
        index=df.index,
    )

    momentum_score += np.select(
        [
            df["RSI_14"] >= 70,
            df["RSI_14"] >= 55,
            df["RSI_14"] >= 45,
        ],
        [
            8,
            7,
            4,
        ],
        default=1,
    )

    momentum_score += (
        df["MACD_BULLISH"].astype(float) * 6
    )

    momentum_score += np.select(
        [
            df["ROC_20"] > 10,
            df["ROC_20"] > 0,
        ],
        [
            6,
            3,
        ],
        default=0,
    )

    # -----------------------------------------------------
    # Trend strength — 15 points
    # -----------------------------------------------------

    strength_score = pd.Series(
        0.0,
        index=df.index,
    )

    strength_score += np.select(
        [
            df["ADX_14"] >= 30,
            df["ADX_14"] >= 25,
            df["ADX_14"] >= 20,
        ],
        [
            8,
            6,
            3,
        ],
        default=0,
    )

    strength_score += (
        (
            df["DI_PLUS"]
            > df["DI_MINUS"]
        ).astype(float)
        * 7
    )

    # -----------------------------------------------------
    # Volume — 10 points
    # -----------------------------------------------------

    volume_score = np.select(
        [
            df["RELATIVE_VOLUME"] >= 2.0,
            df["RELATIVE_VOLUME"] >= 1.5,
            df["RELATIVE_VOLUME"] >= 1.0,
        ],
        [
            10,
            7,
            4,
        ],
        default=0,
    )

    volume_score = pd.Series(
        volume_score,
        index=df.index,
    )

    # -----------------------------------------------------
    # Price structure — 15 points
    # -----------------------------------------------------

    structure_score = pd.Series(
        0.0,
        index=df.index,
    )

    structure_score += (
        df["BREAKOUT_52W"].astype(float) * 10
    )

    structure_score += np.select(
        [
            df["DIST_FROM_52W_HIGH"] >= -5,
            df["DIST_FROM_52W_HIGH"] >= -10,
            df["DIST_FROM_52W_HIGH"] >= -20,
        ],
        [
            5,
            4,
            2,
        ],
        default=0,
    )

    # -----------------------------------------------------
    # Relative momentum proxy — 10 points
    # -----------------------------------------------------

    relative_score = np.select(
        [
            df["RETURN_60D"] >= 20,
            df["RETURN_60D"] >= 10,
            df["RETURN_60D"] >= 0,
        ],
        [
            10,
            7,
            4,
        ],
        default=0,
    )

    relative_score = pd.Series(
        relative_score,
        index=df.index,
    )

    # -----------------------------------------------------
    # Combine
    # -----------------------------------------------------

    score = (
        trend_score
        + momentum_score
        + strength_score
        + volume_score
        + structure_score
        + relative_score
    )

    df["TECHNICAL_SCORE"] = score.clip(
        lower=0,
        upper=100,
    )

    return df


# =========================================================
# LATEST SNAPSHOT
# =========================================================

def latest_snapshot(
    df: pd.DataFrame,
) -> pd.Series:

    if df.empty:
        raise ValueError(
            "Cannot create snapshot from empty dataframe."
        )

    return df.iloc[-1]


# =========================================================
# PROCESS ONE STOCK
# =========================================================

def process_stock(
    price_file: Path,
) -> Path | None:

    try:

        df = load_price_file(
            price_file
        )

        if df.empty:
            return None

        df = calculate_indicators(df)

        df = calculate_score(df)

        output = (
            INDICATOR_DIR
            / price_file.name
        )

        df.to_parquet(
            output,
            engine="pyarrow",
        )

        return output

    except Exception as exc:

        logger.error(
            "Technical analysis failed for %s: %s",
            price_file.name,
            exc,
        )

        return None


# =========================================================
# PROCESS ALL STOCKS
# =========================================================

def process_all() -> dict:

    files = sorted(
        PRICE_DIR.glob("*.parquet")
    )

    total = len(files)

    successful = 0
    failed = 0

    failures = []

    print()
    print("=" * 70)
    print("INDIA STOCK AGENT — TECHNICAL ENGINE")
    print("=" * 70)
    print()

    for position, file in enumerate(
        files,
        start=1,
    ):

        print(
            f"[{position}/{total}] "
            f"{file.stem}",
            flush=True,
        )

        output = process_stock(file)

        if output:

            successful += 1

        else:

            failed += 1

            failures.append(
                file.name
            )

    print()
    print("=" * 70)
    print("TECHNICAL ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        f"Total      : {total}"
    )

    print(
        f"Successful : {successful}"
    )

    print(
        f"Failed     : {failed}"
    )

    print("=" * 70)

    return {
        "total": total,
        "successful": successful,
        "failed": failed,
        "failures": failures,
    }


# =========================================================
# MAIN
# =========================================================

def main():

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    process_all()


if __name__ == "__main__":
    main()
