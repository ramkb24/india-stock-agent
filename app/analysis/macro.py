import os
import json
import numpy as np
import pandas as pd

MACRO_DIR = "data/macro"
OUTPUT_DIR = "data/market"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_macro(name):
    path = os.path.join(MACRO_DIR, f"{name}.parquet")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing macro file: {path}")

    df = pd.read_parquet(path)

    if "Close" not in df.columns:
        raise ValueError(f"{name}: Close column not found")

    df = df[["Close"]].copy()
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna()
    df = df.sort_index()

    return df


def pct_change(df, periods):
    return df["Close"].pct_change(periods=periods) * 100


def build_signal(value, bullish_threshold, bearish_threshold):
    if value >= bullish_threshold:
        return "BULLISH"
    elif value <= bearish_threshold:
        return "BEARISH"
    return "NEUTRAL"


def analyze_macro():

    assets = {}

    for name in ["USDINR", "BRENT", "GOLD", "US10Y", "VIX"]:
        assets[name] = load_macro(name)

    # ---------------------------------------------------------
    # USD / INR
    # ---------------------------------------------------------

    usdinr = assets["USDINR"]

    usd_5d = pct_change(usdinr, 5).iloc[-1]
    usd_20d = pct_change(usdinr, 20).iloc[-1]
    usd_60d = pct_change(usdinr, 60).iloc[-1]

    # Rising USDINR = INR depreciation
    if usd_20d >= 2:
        fx_signal = "HIGH INR DEPRECIATION PRESSURE"
    elif usd_20d >= 0.75:
        fx_signal = "INR DEPRECIATION PRESSURE"
    elif usd_20d <= -1:
        fx_signal = "INR STRENGTH"
    else:
        fx_signal = "NEUTRAL"

    # ---------------------------------------------------------
    # BRENT
    # ---------------------------------------------------------

    brent = assets["BRENT"]

    brent_5d = pct_change(brent, 5).iloc[-1]
    brent_20d = pct_change(brent, 20).iloc[-1]
    brent_60d = pct_change(brent, 60).iloc[-1]

    if brent_20d >= 10:
        oil_signal = "HIGH OIL PRESSURE"
    elif brent_20d >= 5:
        oil_signal = "RISING OIL"
    elif brent_20d <= -10:
        oil_signal = "FALLING OIL"
    else:
        oil_signal = "NEUTRAL"

    # ---------------------------------------------------------
    # GOLD
    # ---------------------------------------------------------

    gold = assets["GOLD"]

    gold_5d = pct_change(gold, 5).iloc[-1]
    gold_20d = pct_change(gold, 20).iloc[-1]
    gold_60d = pct_change(gold, 60).iloc[-1]

    if gold_20d >= 5:
        gold_signal = "STRONG GOLD MOMENTUM"
    elif gold_20d >= 2:
        gold_signal = "RISING GOLD"
    elif gold_20d <= -5:
        gold_signal = "FALLING GOLD"
    else:
        gold_signal = "NEUTRAL"

    # ---------------------------------------------------------
    # US 10Y
    # ---------------------------------------------------------

    us10y = assets["US10Y"]

    yield_5d = us10y["Close"].diff(5).iloc[-1]
    yield_20d = us10y["Close"].diff(20).iloc[-1]
    yield_60d = us10y["Close"].diff(60).iloc[-1]

    if yield_20d >= 0.25:
        yield_signal = "RISING US YIELDS"
    elif yield_20d <= -0.25:
        yield_signal = "FALLING US YIELDS"
    else:
        yield_signal = "STABLE US YIELDS"

    # ---------------------------------------------------------
    # VIX
    # ---------------------------------------------------------

    vix = assets["VIX"]

    vix_current = vix["Close"].iloc[-1]
    vix_5d = pct_change(vix, 5).iloc[-1]
    vix_20d = pct_change(vix, 20).iloc[-1]

    if vix_current >= 25:
        vix_signal = "HIGH FEAR"
    elif vix_current >= 20:
        vix_signal = "ELEVATED FEAR"
    elif vix_current <= 13:
        vix_signal = "LOW VOLATILITY"
    else:
        vix_signal = "NORMAL VOLATILITY"

    # ---------------------------------------------------------
    # MACRO RISK SCORE
    # ---------------------------------------------------------

    risk_points = 0

    # INR depreciation
    if usd_20d >= 2:
        risk_points += 2
    elif usd_20d >= 0.75:
        risk_points += 1
    elif usd_20d <= -1:
        risk_points -= 1

    # Oil
    if brent_20d >= 10:
        risk_points += 2
    elif brent_20d >= 5:
        risk_points += 1
    elif brent_20d <= -10:
        risk_points -= 1

    # US yields
    if yield_20d >= 0.25:
        risk_points += 1
    elif yield_20d <= -0.25:
        risk_points -= 1

    # VIX
    if vix_current >= 25:
        risk_points += 2
    elif vix_current >= 20:
        risk_points += 1
    elif vix_current <= 13:
        risk_points -= 1

    # ---------------------------------------------------------
    # GOLD IS NOT DIRECTLY TREATED AS "RISK"
    # ---------------------------------------------------------
    # Gold can rise because of inflation, geopolitics,
    # real yields, currency movements, or safe-haven demand.
    # Therefore it is recorded separately rather than blindly
    # adding it to the risk score.

    if risk_points >= 4:
        regime = "HIGH MACRO RISK"
    elif risk_points >= 2:
        regime = "ELEVATED MACRO RISK"
    elif risk_points <= -2:
        regime = "SUPPORTIVE MACRO"
    else:
        regime = "NEUTRAL MACRO"

    result = {
        "macro_regime": regime,
        "macro_risk_points": risk_points,

        "USDINR": {
            "latest": float(usdinr["Close"].iloc[-1]),
            "change_5d_pct": float(usd_5d),
            "change_20d_pct": float(usd_20d),
            "change_60d_pct": float(usd_60d),
            "signal": fx_signal,
        },

        "BRENT": {
            "latest": float(brent["Close"].iloc[-1]),
            "change_5d_pct": float(brent_5d),
            "change_20d_pct": float(brent_20d),
            "change_60d_pct": float(brent_60d),
            "signal": oil_signal,
        },

        "GOLD": {
            "latest": float(gold["Close"].iloc[-1]),
            "change_5d_pct": float(gold_5d),
            "change_20d_pct": float(gold_20d),
            "change_60d_pct": float(gold_60d),
            "signal": gold_signal,
        },

        "US10Y": {
            "latest": float(us10y["Close"].iloc[-1]),
            "change_5d_pp": float(yield_5d),
            "change_20d_pp": float(yield_20d),
            "change_60d_pp": float(yield_60d),
            "signal": yield_signal,
        },

        "VIX": {
            "latest": float(vix_current),
            "change_5d_pct": float(vix_5d),
            "change_20d_pct": float(vix_20d),
            "signal": vix_signal,
        },
    }

    return result


if __name__ == "__main__":

    print("=" * 70)
    print("MACRO ANALYSIS")
    print("=" * 70)

    result = analyze_macro()

    print(f"\nMACRO REGIME: {result['macro_regime']}")
    print(f"RISK POINTS : {result['macro_risk_points']}")

    print("\nUSD/INR")
    print(f"  Latest : {result['USDINR']['latest']:.4f}")
    print(f"  5D     : {result['USDINR']['change_5d_pct']:+.2f}%")
    print(f"  20D    : {result['USDINR']['change_20d_pct']:+.2f}%")
    print(f"  60D    : {result['USDINR']['change_60d_pct']:+.2f}%")
    print(f"  Signal : {result['USDINR']['signal']}")

    print("\nBRENT")
    print(f"  Latest : {result['BRENT']['latest']:.2f}")
    print(f"  5D     : {result['BRENT']['change_5d_pct']:+.2f}%")
    print(f"  20D    : {result['BRENT']['change_20d_pct']:+.2f}%")
    print(f"  60D    : {result['BRENT']['change_60d_pct']:+.2f}%")
    print(f"  Signal : {result['BRENT']['signal']}")

    print("\nGOLD")
    print(f"  Latest : {result['GOLD']['latest']:.2f}")
    print(f"  5D     : {result['GOLD']['change_5d_pct']:+.2f}%")
    print(f"  20D    : {result['GOLD']['change_20d_pct']:+.2f}%")
    print(f"  60D    : {result['GOLD']['change_60d_pct']:+.2f}%")
    print(f"  Signal : {result['GOLD']['signal']}")

    print("\nUS 10Y")
    print(f"  Latest : {result['US10Y']['latest']:.3f}%")
    print(f"  5D     : {result['US10Y']['change_5d_pp']:+.3f} pp")
    print(f"  20D    : {result['US10Y']['change_20d_pp']:+.3f} pp")
    print(f"  60D    : {result['US10Y']['change_60d_pp']:+.3f} pp")
    print(f"  Signal : {result['US10Y']['signal']}")

    print("\nVIX")
    print(f"  Latest : {result['VIX']['latest']:.2f}")
    print(f"  5D     : {result['VIX']['change_5d_pct']:+.2f}%")
    print(f"  20D    : {result['VIX']['change_20d_pct']:+.2f}%")
    print(f"  Signal : {result['VIX']['signal']}")

    output = os.path.join(OUTPUT_DIR, "macro_analysis.json")

    with open(output, "w") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 70)
    print("MACRO ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Saved: {output}")
