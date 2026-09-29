from __future__ import annotations

import io
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

NIFTY500_URL = (
    "https://www.niftyindices.com/IndexConstituent/"
    "ind_nifty500list.csv"
)

SENSEX_URL = "https://www.bseindices.com/constituents/code/16"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/140.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def normalize_name(value: object) -> str:
    """Normalize company names for matching."""
    if pd.isna(value):
        return ""

    value = str(value).strip().upper()

    replacements = {
        "&": "AND",
        ".": "",
        ",": "",
        "'": "",
        '"': "",
        "-": " ",
    }

    for old, new in replacements.items():
        value = value.replace(old, new)

    return " ".join(value.split())


def normalize_nse_symbol(symbol: object) -> str:
    """Convert an NSE symbol into a Yahoo-compatible symbol."""
    if pd.isna(symbol):
        return ""

    symbol = str(symbol).strip().upper()

    # Yahoo Finance NSE suffix
    return f"{symbol}.NS"


def normalize_bse_code(code: object) -> str:
    """Convert a BSE numeric code into a Yahoo-compatible symbol."""
    if pd.isna(code):
        return ""

    code = str(code).strip()

    # Remove decimal representation such as 500112.0
    if code.endswith(".0"):
        code = code[:-2]

    return f"{code}.BO"


# ---------------------------------------------------------
# NIFTY 500
# ---------------------------------------------------------

def fetch_nifty500() -> pd.DataFrame:
    """
    Download the current Nifty 500 constituent list
    from the official NSE Indices CSV.
    """

    logger.info("Downloading Nifty 500 constituents...")

    response = requests.get(
        NIFTY500_URL,
        headers=HEADERS,
        timeout=30,
    )

    response.raise_for_status()

    # Nifty's CSV is usually UTF-8/BOM encoded.
    df = pd.read_csv(
        io.BytesIO(response.content),
        encoding="utf-8-sig",
    )

    df.columns = [str(c).strip() for c in df.columns]

    required = {"Company Name", "Symbol"}

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Nifty 500 CSV missing columns: {sorted(missing)}. "
            f"Received columns: {list(df.columns)}"
        )

    result = pd.DataFrame(
        {
            "company_name": df["Company Name"].astype(str).str.strip(),
            "nse_symbol": df["Symbol"].astype(str).str.strip().str.upper(),
            "bse_code": None,
            "exchange": "NSE",
            "index": "NIFTY500",
            "sector": (
                df["Industry"]
                if "Industry" in df.columns
                else None
            ),
        }
    )

    result["yahoo_symbol"] = result["nse_symbol"].map(
        normalize_nse_symbol
    )

    return result


# ---------------------------------------------------------
# SENSEX
# ---------------------------------------------------------

# Sensex constituent seed.
#
# This is intentionally kept separate from the NSE download.
# The BSE web page is not a stable machine-readable endpoint,
# so the universe engine should not silently fail when BSE
# changes its HTML structure.
#
# We will add a proper BSE/API refresh mechanism later.

SENSEX_SEED = [
    ("Adani Ports and Special Economic Zone Ltd.", "ADANIPORTS"),
    ("Asian Paints Ltd.", "ASIANPAINT"),
    ("Axis Bank Ltd.", "AXISBANK"),
    ("Bajaj Finance Ltd.", "BAJFINANCE"),
    ("Bajaj Finserv Ltd.", "BAJAJFINSV"),
    ("Bharat Electronics Ltd.", "BEL"),
    ("Bharti Airtel Ltd.", "BHARTIARTL"),
    ("Eternal Ltd.", "ETERNAL"),
    ("HCL Technologies Ltd.", "HCLTECH"),
    ("HDFC Bank Ltd.", "HDFCBANK"),
    ("Hindustan Unilever Ltd.", "HINDUNILVR"),
    ("ICICI Bank Ltd.", "ICICIBANK"),
    ("Infosys Ltd.", "INFY"),
    ("InterGlobe Aviation Ltd.", "INDIGO"),
    ("ITC Ltd.", "ITC"),
    ("Kotak Mahindra Bank Ltd.", "KOTAKBANK"),
    ("Larsen & Toubro Ltd.", "LT"),
    ("Mahindra & Mahindra Ltd.", "M&M"),
    ("Maruti Suzuki India Ltd.", "MARUTI"),
    ("NTPC Ltd.", "NTPC"),
    ("Power Grid Corporation of India Ltd.", "POWERGRID"),
    ("Reliance Industries Ltd.", "RELIANCE"),
    ("State Bank of India", "SBIN"),
    ("Sun Pharmaceutical Industries Ltd.", "SUNPHARMA"),
    ("Tata Consultancy Services Ltd.", "TCS"),
    ("Tata Motors Ltd.", "TATAMOTORS"),
    ("Tata Steel Ltd.", "TATASTEEL"),
    ("Tech Mahindra Ltd.", "TECHM"),
    ("Titan Company Ltd.", "TITAN"),
    ("Trent Ltd.", "TRENT"),
]


def fetch_sensex() -> pd.DataFrame:
    """
    Build the current Sensex constituent table from the
    maintained seed list.

    The seed is deliberately isolated so it can later be
    replaced by an official BSE API/data-feed implementation
    without changing the rest of the universe engine.
    """

    logger.info("Loading Sensex constituents...")

    rows = []

    for company_name, nse_symbol in SENSEX_SEED:
        rows.append(
            {
                "company_name": company_name,
                "nse_symbol": nse_symbol,
                "bse_code": None,
                "exchange": "BSE",
                "index": "SENSEX",
                "sector": None,
                "yahoo_symbol": normalize_nse_symbol(nse_symbol),
            }
        )

    return pd.DataFrame(rows)

# ---------------------------------------------------------
# Combine universe
# ---------------------------------------------------------

def build_universe() -> pd.DataFrame:
    """
    Build the combined Nifty 500 + Sensex universe.

    A company can belong to both indices, so duplicates are
    consolidated into a single company record.
    """

    nifty = fetch_nifty500()

    try:
        sensex = fetch_sensex()
    except Exception as exc:
        logger.warning(
            "Sensex download failed: %s",
            exc,
        )

        # Continue with Nifty 500 rather than losing the
        # entire universe.
        sensex = pd.DataFrame(
            columns=nifty.columns
        )

    combined = pd.concat(
        [nifty, sensex],
        ignore_index=True,
    )

    combined["company_key"] = combined["company_name"].map(
        normalize_name
    )

    # A company appearing in both indexes should become one
    # universe record.
    grouped = []

    for company_key, group in combined.groupby(
        "company_key",
        sort=False,
    ):
        first = group.iloc[0].copy()

        indices = sorted(
            set(group["index"].dropna().astype(str))
        )

        exchanges = sorted(
            set(group["exchange"].dropna().astype(str))
        )

        first["index"] = ",".join(indices)
        first["exchange"] = ",".join(exchanges)

        # Prefer NSE symbol if one exists.
        nse_symbols = (
            group["nse_symbol"]
            .dropna()
            .astype(str)
            .loc[lambda x: x.ne("None")]
            .tolist()
        )

        if nse_symbols:
            first["nse_symbol"] = nse_symbols[0]
            first["yahoo_symbol"] = normalize_nse_symbol(
                nse_symbols[0]
            )

        bse_codes = (
            group["bse_code"]
            .dropna()
            .astype(str)
            .loc[lambda x: x.ne("None")]
            .tolist()
        )

        if bse_codes:
            first["bse_code"] = bse_codes[0]

        grouped.append(first)

    universe = pd.DataFrame(grouped)

    universe = universe[
        [
            "company_name",
            "company_key",
            "nse_symbol",
            "bse_code",
            "yahoo_symbol",
            "exchange",
            "index",
            "sector",
        ]
    ]

    universe = universe.sort_values(
        "company_name"
    ).reset_index(drop=True)

    # Index adjustment placeholders such as DUMMYHEG are
    # legitimate index constituents for index accounting,
    # but they are not currently tradable securities.
    universe["tradable"] = ~universe["nse_symbol"].astype(str).str.startswith(
        "DUMMY"
    )

    universe["last_updated"] = datetime.now(
        timezone.utc
    ).isoformat()

    return universe


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

def save_universe(
    universe: pd.DataFrame,
) -> Path:

    output = DATA_DIR / "universe.csv"

    universe.to_csv(
        output,
        index=False,
    )

    logger.info(
        "Saved %s securities to %s",
        len(universe),
        output,
    )

    return output


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def main() -> None:

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    universe = build_universe()

    output = save_universe(universe)

    print()
    print("=" * 70)
    print("INDIA STOCK AGENT — UNIVERSE")
    print("=" * 70)
    print(f"Total unique companies : {len(universe)}")
    print(
        "Nifty 500 members      : "
        f"{universe['index'].str.contains('NIFTY500').sum()}"
    )
    print(
        "Sensex members         : "
        f"{universe['index'].str.contains('SENSEX').sum()}"
    )
    print(f"Saved to               : {output}")
    print("=" * 70)
    print()

    print(
        universe[
            [
                "company_name",
                "nse_symbol",
                "bse_code",
                "index",
            ]
        ].head(20).to_string(index=False)
    )


if __name__ == "__main__":
    main()
