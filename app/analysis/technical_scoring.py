"""
Single source of truth for the technical scoring function.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_technical_components(df: pd.DataFrame) -> pd.DataFrame:
    close = pd.to_numeric(df["Close"], errors="coerce")
    high = pd.to_numeric(df["High"], errors="coerce")
    low = pd.to_numeric(df["Low"], errors="coerce")
    volume = pd.to_numeric(df["Volume"], errors="coerce")

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma100 = close.rolling(100).mean()
    sma200 = close.rolling(200).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(
        np.where((up > down) & (up > 0), up, 0.0), index=df.index
    )
    minus_dm = pd.Series(
        np.where((down > up) & (down > 0), down, 0.0), index=df.index
    )
    tr14 = tr.rolling(14).sum()
    plus_di = 100 * plus_dm.rolling(14).sum() / tr14.replace(0, np.nan)
    minus_di = 100 * minus_dm.rolling(14).sum() / tr14.replace(0, np.nan)
    dx = (
        100
        * (plus_di - minus_di).abs()
        / (plus_di + minus_di).replace(0, np.nan)
    )
    adx = dx.rolling(14).mean()

    ret20 = close.pct_change(20) * 100
    ret60 = close.pct_change(60) * 100
    ret120 = close.pct_change(120) * 100

    vol_sma20 = volume.rolling(20).mean()
    rel_vol = volume / vol_sma20.replace(0, np.nan)

    prev_252_high = close.shift(1).rolling(252).max()
    prev_252_low = close.shift(1).rolling(252).min()

    above20 = close > sma20
    above50 = close > sma50
    above100 = close > sma100
    above200 = close > sma200

    bullish_align = (sma20 > sma50) & (sma50 > sma100) & (sma100 > sma200)
    bearish_align = (sma20 < sma50) & (sma50 < sma100) & (sma100 < sma200)

    macd_bullish = macd > macd_signal
    strong_trend = adx >= 25
    adx_bullish = (adx >= 20) & (plus_di > minus_di)

    breakout_52w = close > prev_252_high
    breakdown_52w = close < prev_252_low

    high_vol = rel_vol >= 1.25
    very_high_vol = rel_vol >= 1.50

    score = pd.Series(50.0, index=df.index)

    def add(cond, pts):
        cond = cond.fillna(False)
        score.loc[cond] += pts

    add(above20, 5)
    add(above50, 5)
    add(above100, 5)
    add(above200, 5)

    add(bullish_align, 8)
    add(bearish_align, -8)

    add(macd_bullish, 5)
    add(strong_trend, 4)
    add(adx_bullish, 4)

    add(breakout_52w, 8)
    add(breakdown_52w, -8)

    add(high_vol, 2)
    add(very_high_vol, 3)

    score += ret20.clip(-20, 20) * 0.25
    score += ret60.clip(-30, 30) * 0.15
    score += ret120.clip(-50, 50) * 0.10

    score.loc[(rsi >= 50) & (rsi <= 70)] += 5
    score.loc[rsi < 30] -= 3
    score.loc[rsi > 80] -= 5

    tiebreak = (
        ret120.fillna(0.0) * 0.001
        + (100.0 + (close / prev_252_high - 1) * 100).fillna(0.0) * 0.0001
    )

    return pd.DataFrame(
        {
            "SCORE_RAW": score,
            "TIEBREAK": tiebreak,
            "RSI_14": rsi,
            "MACD": macd,
            "MACD_SIGNAL": macd_signal,
            "ADX_14": adx,
            "RETURN_20D": ret20,
            "RETURN_60D": ret60,
            "RETURN_120D": ret120,
            "DIST_FROM_52W_HIGH": (close / prev_252_high - 1) * 100,
            "ABOVE_SMA_20": above20,
            "ABOVE_SMA_50": above50,
            "ABOVE_SMA_100": above100,
            "ABOVE_SMA_200": above200,
            "BULLISH_ALIGNMENT": bullish_align,
            "BEARISH_ALIGNMENT": bearish_align,
            "BREAKOUT_52W": breakout_52w,
            "BREAKDOWN_52W": breakdown_52w,
        },
        index=df.index,
    )


def calculate_technical_score(df: pd.DataFrame) -> pd.Series:
    components = calculate_technical_components(df)
    return components["SCORE_RAW"]
