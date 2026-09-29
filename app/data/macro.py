import os
import time
import json
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf


DATA_DIR = "data/macro"
os.makedirs(DATA_DIR, exist_ok=True)


MACRO_ASSETS = {
    "USDINR": "INR=X",
    "BRENT": "BZ=F",
    "GOLD": "GC=F",
    "US10Y": "^TNX",
    "VIX": "^VIX",
}


def download_macro_history(period="5y"):
    results = {}

    for name, ticker_symbol in MACRO_ASSETS.items():
        print(f"Downloading {name} ({ticker_symbol})")

        try:
            df = yf.download(
                ticker_symbol,
                period=period,
                interval="1d",
                auto_adjust=False,
                progress=False,
            )

            if df.empty:
                print(f"  NO DATA: {name}")
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df.index = pd.to_datetime(df.index)
            df.index.name = "Date"

            output = os.path.join(DATA_DIR, f"{name}.parquet")
            df.to_parquet(output)

            results[name] = {
                "ticker": ticker_symbol,
                "rows": len(df),
                "start": str(df.index.min().date()),
                "end": str(df.index.max().date()),
                "latest_close": float(df["Close"].iloc[-1]),
                "file": output,
            }

            print(
                f"  OK: {len(df)} rows | "
                f"{df.index.min().date()} -> {df.index.max().date()} | "
                f"latest={df['Close'].iloc[-1]:.4f}"
            )

        except Exception as e:
            print(f"  ERROR: {name}: {e}")

        time.sleep(0.5)

    return results


def build_macro_summary(results):
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "assets": results,
    }

    output = os.path.join(DATA_DIR, "macro_summary.json")

    with open(output, "w") as f:
        json.dump(summary, f, indent=2)

    return output


if __name__ == "__main__":
    print("=" * 70)
    print("MACRO DATA DOWNLOAD")
    print("=" * 70)

    results = download_macro_history(period="5y")

    output = build_macro_summary(results)

    print("\n" + "=" * 70)
    print("MACRO DOWNLOAD COMPLETE")
    print("=" * 70)

    print(f"Assets downloaded: {len(results)}")

    for name, info in results.items():
        print(
            f"{name:10} "
            f"{info['rows']:5} rows | "
            f"latest={info['latest_close']:.4f}"
        )

    print(f"\nSaved summary: {output}")
