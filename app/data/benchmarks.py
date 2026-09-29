from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
BENCHMARK_DIR = DATA_DIR / "benchmarks"

BENCHMARK_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

BENCHMARKS = {
    "NIFTY50": "^NSEI",
    "SENSEX": "^BSESN",
    "NIFTY500": "^CRSLDX",
}


def download_benchmark(
    name: str,
    symbol: str,
    period: str = "5y",
) -> pd.DataFrame:

    print(f"Downloading {name} ({symbol})...")

    ticker = yf.Ticker(symbol)

    df = ticker.history(
        period=period,
        interval="1d",
        auto_adjust=True,
        actions=False,
    )

    if df.empty:
        raise ValueError(
            f"No data returned for {name} ({symbol})"
        )

    df = df[
        [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ]
    ]

    df.index = pd.to_datetime(
        df.index,
        errors="coerce",
    )

    df = df[
        ~df.index.isna()
    ]

    df = df[
        ~df.index.duplicated(
            keep="last"
        )
    ]

    df = df.sort_index()

    output = (
        BENCHMARK_DIR
        / f"{name}.parquet"
    )

    df.to_parquet(
        output,
        engine="pyarrow",
    )

    print(
        f"  Saved {len(df)} rows -> {output}"
    )

    return df


def download_all():

    print()
    print("=" * 70)
    print("INDIA STOCK AGENT — BENCHMARK DATA")
    print("=" * 70)
    print()

    results = {}

    for name, symbol in BENCHMARKS.items():

        try:

            df = download_benchmark(
                name,
                symbol,
            )

            results[name] = len(df)

        except Exception as exc:

            print(
                f"ERROR {name}: {exc}"
            )

    print()
    print("=" * 70)
    print("BENCHMARK DOWNLOAD COMPLETE")
    print("=" * 70)

    for name, rows in results.items():

        print(
            f"{name:12} : {rows} rows"
        )

    print("=" * 70)

    return results


def load_benchmark(
    name: str,
) -> pd.DataFrame:

    path = (
        BENCHMARK_DIR
        / f"{name}.parquet"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Benchmark data not found: {path}"
        )

    return pd.read_parquet(path)


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    download_all()
