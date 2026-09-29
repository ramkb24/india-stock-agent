from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PRICE_DIR = DATA_DIR / "prices"

PRICE_DIR.mkdir(parents=True, exist_ok=True)

HISTORY_PERIOD = "5y"
INTERVAL = "1d"


def load_universe() -> pd.DataFrame:
    """Load the master stock universe."""

    path = DATA_DIR / "universe.csv"

    if not path.exists():
        raise FileNotFoundError(
            "data/universe.csv not found. "
            "Run the universe engine first."
        )

    df = pd.read_csv(path)

    required = {
        "company_name",
        "nse_symbol",
        "yahoo_symbol",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Universe missing columns: {sorted(missing)}"
        )

    return df


def clean_price_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize downloaded OHLCV data."""

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    # yfinance can return a MultiIndex depending on the request.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    expected = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    available = [
        column
        for column in expected
        if column in df.columns
    ]

    df = df[available]

    df = df.dropna(
        subset=["Open", "High", "Low", "Close"]
    )

    df.index = pd.to_datetime(
        df.index,
        errors="coerce",
    )

    df = df[~df.index.isna()]

    df = df.sort_index()

    # Remove duplicate trading dates.
    df = df[~df.index.duplicated(keep="last")]

    return df


def download_stock(
    yahoo_symbol: str,
    period: str = HISTORY_PERIOD,
) -> pd.DataFrame:
    """Download historical data for one stock."""

    logger.info(
        "Downloading %s",
        yahoo_symbol,
    )

    try:
        ticker = yf.Ticker(yahoo_symbol)

        df = ticker.history(
            period=period,
            interval=INTERVAL,
            auto_adjust=True,
            actions=False,
        )

        return clean_price_data(df)

    except Exception as exc:
        logger.error(
            "Failed %s: %s",
            yahoo_symbol,
            exc,
        )

        return pd.DataFrame()


def save_stock_data(
    yahoo_symbol: str,
    df: pd.DataFrame,
) -> Path | None:
    """Save one stock's data as Parquet."""

    if df.empty:
        return None

    safe_name = (
        yahoo_symbol
        .replace(".", "_")
        .replace("/", "_")
    )

    path = PRICE_DIR / f"{safe_name}.parquet"

    df.to_parquet(
        path,
        engine="pyarrow",
    )

    return path


def download_all(
    limit: int | None = None,
    delay: float = 0.25,
) -> dict:

    universe = load_universe()

    if limit:
        universe = universe.head(limit)

    total = len(universe)

    successful = 0
    failed = 0
    empty = 0

    failures = []

    logger.info(
        "Starting download for %s stocks",
        total,
    )

    for position, row in enumerate(
        universe.itertuples(index=False),
        start=1,
    ):

        symbol = row.yahoo_symbol

        print(
            f"[{position}/{total}] {symbol}",
            flush=True,
        )

        if not isinstance(symbol, str) or not symbol:
            failed += 1
            failures.append(
                {
                    "company": row.company_name,
                    "symbol": symbol,
                    "reason": "Invalid symbol",
                }
            )
            continue

        df = download_stock(symbol)

        if df.empty:
            empty += 1

            failures.append(
                {
                    "company": row.company_name,
                    "symbol": symbol,
                    "reason": "No data",
                }
            )

            continue

        try:
            save_stock_data(
                symbol,
                df,
            )

            successful += 1

        except Exception as exc:
            failed += 1

            failures.append(
                {
                    "company": row.company_name,
                    "symbol": symbol,
                    "reason": str(exc),
                }
            )

        time.sleep(delay)

    result = {
        "total": total,
        "successful": successful,
        "empty": empty,
        "failed": failed,
        "failures": failures,
    }

    return result


def load_stock(
    yahoo_symbol: str,
) -> pd.DataFrame:
    """Load locally stored price data."""

    safe_name = (
        yahoo_symbol
        .replace(".", "_")
        .replace("/", "_")
    )

    path = PRICE_DIR / f"{safe_name}.parquet"

    if not path.exists():
        raise FileNotFoundError(
            f"No local data for {yahoo_symbol}"
        )

    return pd.read_parquet(path)


def main() -> None:

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    print()
    print("=" * 70)
    print("INDIA STOCK AGENT — MARKET DATA ENGINE")
    print("=" * 70)
    print()

    result = download_all()

    print()
    print("=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)

    print(
        f"Total       : {result['total']}"
    )

    print(
        f"Successful  : {result['successful']}"
    )

    print(
        f"No data     : {result['empty']}"
    )

    print(
        f"Failed      : {result['failed']}"
    )

    print("=" * 70)

    if result["failures"]:

        print()
        print("FAILED / EMPTY SYMBOLS")
        print("-" * 70)

        for failure in result["failures"][:50]:

            print(
                f"{failure['symbol']} | "
                f"{failure['company']} | "
                f"{failure['reason']}"
            )


if __name__ == "__main__":
    main()
