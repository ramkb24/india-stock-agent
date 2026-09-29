import os
import json
import numpy as np
import pandas as pd

PRICE_DIR = "data/prices"
OUTPUT_DIR = "data/market"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_price(symbol):
    path = os.path.join(PRICE_DIR, f"{symbol}.parquet")

    if not os.path.exists(path):
        return None

    df = pd.read_parquet(path)

    if "Close" not in df.columns:
        return None

    df = df.copy()

    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")

    df = df.dropna(subset=["Close"])
    df = df.sort_index()

    return df


def annualized_volatility(returns):
    if len(returns) < 20:
        return np.nan

    return returns.std() * np.sqrt(252) * 100


def max_drawdown(prices):
    if len(prices) < 2:
        return np.nan

    running_max = prices.cummax()
    drawdown = (prices / running_max - 1) * 100

    return drawdown.min()


def downside_volatility(returns):
    negative = returns[returns < 0]

    if len(negative) < 10:
        return np.nan

    return negative.std() * np.sqrt(252) * 100


def calculate_risk(symbol):

    df = load_price(symbol)

    if df is None or len(df) < 60:
        return None

    close = df["Close"]

    returns = close.pct_change().dropna()

    latest = float(close.iloc[-1])

    # ---------------------------------------------------------
    # VOLATILITY
    # ---------------------------------------------------------

    vol_20d = annualized_volatility(returns.tail(20))
    vol_60d = annualized_volatility(returns.tail(60))
    vol_1y = annualized_volatility(returns.tail(252))

    downside_vol = downside_volatility(returns.tail(252))

    # ---------------------------------------------------------
    # DRAWDOWN
    # ---------------------------------------------------------

    max_dd_1y = max_drawdown(close.tail(252))

    rolling_high_20 = close.tail(20).max()
    recent_drawdown_20 = (latest / rolling_high_20 - 1) * 100

    rolling_high_60 = close.tail(60).max()
    recent_drawdown_60 = (latest / rolling_high_60 - 1) * 100

    # ---------------------------------------------------------
    # 52 WEEK POSITION
    # ---------------------------------------------------------

    high_52w = close.tail(252).max()
    low_52w = close.tail(252).min()

    distance_from_high = (latest / high_52w - 1) * 100
    distance_from_low = (latest / low_52w - 1) * 100

    # ---------------------------------------------------------
    # RETURN PROFILE
    # ---------------------------------------------------------

    return_20d = (
        (latest / close.iloc[-21] - 1) * 100
        if len(close) >= 21
        else np.nan
    )

    return_60d = (
        (latest / close.iloc[-61] - 1) * 100
        if len(close) >= 61
        else np.nan
    )

    return_252d = (
        (latest / close.iloc[-253] - 1) * 100
        if len(close) >= 253
        else np.nan
    )

    # ---------------------------------------------------------
    # LIQUIDITY
    # ---------------------------------------------------------

    volume = df["Volume"].dropna()

    avg_volume_20 = (
        float(volume.tail(20).mean())
        if len(volume) >= 20
        else np.nan
    )

    median_volume_20 = (
        float(volume.tail(20).median())
        if len(volume) >= 20
        else np.nan
    )

    # ---------------------------------------------------------
    # RISK SCORE
    # Higher = more risk
    # ---------------------------------------------------------

    risk_points = 0

    # Volatility
    if pd.notna(vol_1y):
        if vol_1y >= 60:
            risk_points += 4
        elif vol_1y >= 45:
            risk_points += 3
        elif vol_1y >= 30:
            risk_points += 2
        elif vol_1y >= 20:
            risk_points += 1

    # Recent volatility
    if pd.notna(vol_20d) and pd.notna(vol_1y):
        if vol_20d > vol_1y * 1.5:
            risk_points += 2
        elif vol_20d > vol_1y * 1.25:
            risk_points += 1

    # Maximum drawdown
    if pd.notna(max_dd_1y):
        if max_dd_1y <= -50:
            risk_points += 4
        elif max_dd_1y <= -35:
            risk_points += 3
        elif max_dd_1y <= -25:
            risk_points += 2
        elif max_dd_1y <= -15:
            risk_points += 1

    # Recent drawdown
    if recent_drawdown_20 <= -20:
        risk_points += 2
    elif recent_drawdown_20 <= -10:
        risk_points += 1

    # Distance from 52-week high
    if distance_from_high <= -40:
        risk_points += 3
    elif distance_from_high <= -25:
        risk_points += 2
    elif distance_from_high <= -15:
        risk_points += 1

    # Downside volatility
    if pd.notna(downside_vol):
        if downside_vol >= 45:
            risk_points += 3
        elif downside_vol >= 30:
            risk_points += 2
        elif downside_vol >= 20:
            risk_points += 1

    # ---------------------------------------------------------
    # CLASSIFICATION
    # ---------------------------------------------------------

    if risk_points >= 12:
        risk_class = "VERY_HIGH"
    elif risk_points >= 8:
        risk_class = "HIGH"
    elif risk_points >= 4:
        risk_class = "MODERATE"
    else:
        risk_class = "LOW"

    return {
        "symbol": symbol,
        "latest_price": latest,

        "volatility_20d": vol_20d,
        "volatility_60d": vol_60d,
        "volatility_1y": vol_1y,
        "downside_volatility": downside_vol,

        "max_drawdown_1y": max_dd_1y,
        "recent_drawdown_20d": recent_drawdown_20,
        "recent_drawdown_60d": recent_drawdown_60,

        "high_52w": high_52w,
        "low_52w": low_52w,
        "distance_from_52w_high": distance_from_high,
        "distance_from_52w_low": distance_from_low,

        "return_20d": return_20d,
        "return_60d": return_60d,
        "return_252d": return_252d,

        "average_volume_20d": avg_volume_20,
        "median_volume_20d": median_volume_20,

        "risk_points": risk_points,
        "risk_class": risk_class,
    }


def run():

    files = [
        f for f in os.listdir(PRICE_DIR)
        if f.endswith(".parquet")
    ]

    results = []

    print("=" * 70)
    print("STOCK RISK ENGINE")
    print("=" * 70)

    for i, filename in enumerate(sorted(files), 1):

        symbol = filename.replace(".parquet", "")

        result = calculate_risk(symbol)

        if result is not None:
            results.append(result)

        if i % 50 == 0:
            print(f"Processed {i}/{len(files)}")

    df = pd.DataFrame(results)

    # Save detailed CSV
    csv_path = os.path.join(
        OUTPUT_DIR,
        "stock_risk.csv"
    )

    df.to_csv(csv_path, index=False)

    # Summary
    class_counts = (
        df["risk_class"]
        .value_counts()
        .to_dict()
    )

    summary = {
        "stocks_analyzed": int(len(df)),
        "risk_class_counts": class_counts,
        "generated_at": pd.Timestamp.utcnow().isoformat(),
    }

    json_path = os.path.join(
        OUTPUT_DIR,
        "risk_summary.json"
    )

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("RISK ENGINE COMPLETE")
    print("=" * 70)

    print(f"Stocks analyzed: {len(df)}")

    print("\nRisk classification:")
    for risk_class, count in class_counts.items():
        print(f"  {risk_class:12} {count}")

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")

    print("\nHighest risk examples:")

    top_risk = (
        df.sort_values(
            ["risk_points", "volatility_1y"],
            ascending=False
        )
        .head(15)
    )

    columns = [
        "symbol",
        "risk_points",
        "risk_class",
        "volatility_1y",
        "max_drawdown_1y",
        "distance_from_52w_high",
    ]

    print(
        top_risk[columns]
        .to_string(index=False)
    )


if __name__ == "__main__":
    run()
