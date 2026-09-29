import os
import pandas as pd
import numpy as np


UNIVERSE_FILE = "data/universe.csv"
INDICATOR_DIR = "data/indicators"
OUTPUT_FILE = "data/market/sector_analysis.json"
STOCK_OUTPUT_FILE = "data/market/stock_sector_relative_strength.json"


def load_data():

    universe = pd.read_csv(UNIVERSE_FILE)

    rows = []

    for _, stock in universe.iterrows():

        if not bool(stock.get("tradable", True)):
            continue

        symbol = str(stock["nse_symbol"])
        sector = str(stock["sector"])

        path = os.path.join(
            INDICATOR_DIR,
            f"{symbol.replace('.', '_')}_NS.parquet"
        )

        if not os.path.exists(path):
            continue

        try:

            df = pd.read_parquet(path)

            if df.empty:
                continue

            latest = df.iloc[-1]

            rows.append({

                "symbol": symbol,
                "company_name": stock["company_name"],
                "sector": sector,

                "technical_score": latest["TECHNICAL_SCORE"],
                "rsi": latest["RSI_14"],

                "return_20d": latest["RETURN_20D"],
                "return_60d": latest["RETURN_60D"],
                "return_120d": latest["RETURN_120D"],

                "above_sma20": latest["ABOVE_SMA_20"],
                "above_sma50": latest["ABOVE_SMA_50"],
                "above_sma200": latest["ABOVE_SMA_200"],

                "breakout_52w": latest["BREAKOUT_52W"],
                "breakdown_52w": latest["BREAKDOWN_52W"],

            })

        except Exception as e:

            print(f"WARNING: {symbol}: {e}")

    return pd.DataFrame(rows)


def build_sector_analysis(df):

    results = []

    for sector, group in df.groupby("sector"):

        n = len(group)

        technical_score = pd.to_numeric(
            group["technical_score"],
            errors="coerce"
        )

        rsi = pd.to_numeric(
            group["rsi"],
            errors="coerce"
        )

        return_20d = pd.to_numeric(
            group["return_20d"],
            errors="coerce"
        )

        return_60d = pd.to_numeric(
            group["return_60d"],
            errors="coerce"
        )

        return_120d = pd.to_numeric(
            group["return_120d"],
            errors="coerce"
        )

        above20 = group["above_sma20"].astype(bool).mean() * 100
        above50 = group["above_sma50"].astype(bool).mean() * 100
        above200 = group["above_sma200"].astype(bool).mean() * 100

        breakout_rate = group["breakout_52w"].astype(bool).mean() * 100
        breakdown_rate = group["breakdown_52w"].astype(bool).mean() * 100

        avg_score = technical_score.mean()
        avg_rsi = rsi.mean()

        avg_return_20d = return_20d.mean()
        avg_return_60d = return_60d.mean()
        avg_return_120d = return_120d.mean()

        breadth_score = (
            above20 * 0.25
            + above50 * 0.35
            + above200 * 0.40
        )

        momentum_score = (
            avg_return_20d * 0.30
            + avg_return_60d * 0.35
            + avg_return_120d * 0.35
        )

        results.append({

            "sector": sector,
            "stocks": n,

            "avg_technical_score": round(avg_score, 2),
            "avg_rsi": round(avg_rsi, 2),

            "avg_return_20d_pct": round(avg_return_20d, 2),
            "avg_return_60d_pct": round(avg_return_60d, 2),
            "avg_return_120d_pct": round(avg_return_120d, 2),

            "above_20dma_pct": round(above20, 2),
            "above_50dma_pct": round(above50, 2),
            "above_200dma_pct": round(above200, 2),

            "breakout_52w_pct": round(breakout_rate, 2),
            "breakdown_52w_pct": round(breakdown_rate, 2),

            "breadth_score": round(breadth_score, 2),
            "momentum_score": round(momentum_score, 2),
        })

    result = pd.DataFrame(results)

    min_m = result["momentum_score"].min()
    max_m = result["momentum_score"].max()

    if max_m > min_m:

        result["momentum_percentile"] = (
            (result["momentum_score"] - min_m)
            / (max_m - min_m)
            * 100
        )

    else:

        result["momentum_percentile"] = 50

    result["sector_score"] = (
        result["breadth_score"] * 0.50
        + result["momentum_percentile"] * 0.50
    )

    result["sector_score"] = result["sector_score"].clip(0, 100)

    result = result.sort_values(
        "sector_score",
        ascending=False
    ).reset_index(drop=True)

    return result


def build_relative_strength(df, sector_df):

    sector_lookup = sector_df.set_index("sector")

    result = df.copy()

    result["sector_avg_return_20d"] = (
        result["sector"]
        .map(sector_lookup["avg_return_20d_pct"])
    )

    result["sector_avg_return_60d"] = (
        result["sector"]
        .map(sector_lookup["avg_return_60d_pct"])
    )

    result["sector_avg_return_120d"] = (
        result["sector"]
        .map(sector_lookup["avg_return_120d_pct"])
    )

    result["sector_avg_technical_score"] = (
        result["sector"]
        .map(sector_lookup["avg_technical_score"])
    )

    # ------------------------------------------
    # RELATIVE RETURNS
    # ------------------------------------------

    result["relative_strength_20d"] = (
        result["return_20d"]
        - result["sector_avg_return_20d"]
    )

    result["relative_strength_60d"] = (
        result["return_60d"]
        - result["sector_avg_return_60d"]
    )

    result["relative_strength_120d"] = (
        result["return_120d"]
        - result["sector_avg_return_120d"]
    )

    # ------------------------------------------
    # RELATIVE TECHNICAL SCORE
    # ------------------------------------------

    result["relative_technical_score"] = (
        result["technical_score"]
        - result["sector_avg_technical_score"]
    )

    # ------------------------------------------
    # SECTOR PERCENTILES
    # ------------------------------------------

    result["rs_20d_percentile"] = (
        result
        .groupby("sector")["relative_strength_20d"]
        .rank(pct=True)
        * 100
    )

    result["rs_60d_percentile"] = (
        result
        .groupby("sector")["relative_strength_60d"]
        .rank(pct=True)
        * 100
    )

    result["rs_120d_percentile"] = (
        result
        .groupby("sector")["relative_strength_120d"]
        .rank(pct=True)
        * 100
    )

    result["relative_strength_score"] = (
        result["rs_20d_percentile"] * 0.30
        + result["rs_60d_percentile"] * 0.35
        + result["rs_120d_percentile"] * 0.35
    )

    # ------------------------------------------
    # STOCK VS SECTOR TECHNICAL STRENGTH
    # ------------------------------------------

    result["technical_vs_sector"] = (
        result["technical_score"]
        - result["sector_avg_technical_score"]
    )

    # ------------------------------------------
    # SAVE
    # ------------------------------------------

    result = result.sort_values(
        "relative_strength_score",
        ascending=False
    ).reset_index(drop=True)

    return result


def main():

    print("=" * 70)
    print("SECTOR + STOCK RELATIVE STRENGTH ENGINE")
    print("=" * 70)

    df = load_data()

    print()
    print("Stocks analyzed:", len(df))

    sector_df = build_sector_analysis(df)

    stock_df = build_relative_strength(
        df,
        sector_df
    )

    os.makedirs("data/market", exist_ok=True)

    sector_df.to_json(
        OUTPUT_FILE,
        orient="records",
        indent=2
    )

    stock_df.to_json(
        STOCK_OUTPUT_FILE,
        orient="records",
        indent=2
    )

    print()
    print("Sectors:", len(sector_df))
    print("Stocks:", len(stock_df))

    print()
    print("=" * 70)
    print("TOP RELATIVE-STRENGTH STOCKS")
    print("=" * 70)

    columns = [
        "symbol",
        "company_name",
        "sector",
        "technical_score",
        "relative_strength_score",
        "relative_strength_20d",
        "relative_strength_60d",
        "relative_strength_120d",
        "technical_vs_sector",
    ]

    print(
        stock_df[columns]
        .head(30)
        .to_string(index=False)
    )

    print()
    print("=" * 70)
    print("BOTTOM RELATIVE-STRENGTH STOCKS")
    print("=" * 70)

    print(
        stock_df[columns]
        .tail(20)
        .to_string(index=False)
    )

    print()
    print("Saved:")
    print(OUTPUT_FILE)
    print(STOCK_OUTPUT_FILE)


if __name__ == "__main__":
    main()
