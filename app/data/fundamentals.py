import os
import time
import json
import numpy as np
import pandas as pd
import yfinance as yf


UNIVERSE_FILE = "data/universe.csv"
OUTPUT_FILE = "data/fundamentals.csv"
RAW_DIR = "data/fundamentals_raw"


SNAPSHOT_FIELDS = [
    "marketCap",
    "trailingPE",
    "forwardPE",
    "priceToBook",
    "enterpriseToEbitda",
    "profitMargins",
    "operatingMargins",
    "returnOnEquity",
    "returnOnAssets",
    "debtToEquity",
    "currentRatio",
    "revenueGrowth",
    "earningsGrowth",
    "dividendYield",
]


def safe_float(value):
    try:
        if value is None:
            return np.nan

        value = float(value)

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return np.nan


def get_statement_value(statement, row_name, column):
    try:
        if row_name not in statement.index:
            return np.nan

        value = statement.loc[row_name, column]

        return safe_float(value)

    except Exception:
        return np.nan


def calculate_growth(values):
    """
    Calculate CAGR from the oldest valid observation
    to the newest valid observation.
    """

    values = pd.Series(values).dropna()

    if len(values) < 2:
        return np.nan

    oldest = safe_float(values.iloc[-1])
    newest = safe_float(values.iloc[0])

    if not np.isfinite(oldest) or not np.isfinite(newest):
        return np.nan

    if oldest <= 0 or newest <= 0:
        return np.nan

    years = len(values) - 1

    if years <= 0:
        return np.nan

    return ((newest / oldest) ** (1 / years) - 1) * 100


def process_stock(stock):

    symbol = str(stock["nse_symbol"])
    yahoo_symbol = f"{symbol}.NS"

    sector = str(stock["sector"])

    print(f"Processing {symbol}")

    ticker = yf.Ticker(yahoo_symbol)

    result = {
        "symbol": symbol,
        "company_name": stock["company_name"],
        "sector": sector,

        "market_cap": np.nan,

        "trailing_pe": np.nan,
        "forward_pe": np.nan,
        "price_to_book": np.nan,
        "enterprise_to_ebitda": np.nan,

        "profit_margin": np.nan,
        "operating_margin": np.nan,

        "roe": np.nan,
        "roa": np.nan,

        "debt_to_equity": np.nan,
        "current_ratio": np.nan,

        "snapshot_revenue_growth": np.nan,
        "snapshot_earnings_growth": np.nan,
        "dividend_yield": np.nan,

        "revenue_cagr": np.nan,
        "net_income_cagr": np.nan,
        "eps_cagr": np.nan,

        "operating_margin_latest": np.nan,
        "operating_margin_oldest": np.nan,
        "operating_margin_change": np.nan,

        "calculated_roe": np.nan,
        "calculated_roa": np.nan,
        "calculated_debt_to_equity": np.nan,
        "calculated_current_ratio": np.nan,

        "financial_years": 0,
        "fundamental_data_quality": "NONE",
    }

    # --------------------------------------------------
    # SNAPSHOT
    # --------------------------------------------------

    try:

        info = ticker.info

        for field in SNAPSHOT_FIELDS:

            key = field

            if field == "marketCap":
                output_key = "market_cap"

            elif field == "trailingPE":
                output_key = "trailing_pe"

            elif field == "forwardPE":
                output_key = "forward_pe"

            elif field == "priceToBook":
                output_key = "price_to_book"

            elif field == "enterpriseToEbitda":
                output_key = "enterprise_to_ebitda"

            elif field == "profitMargins":
                output_key = "profit_margin"

            elif field == "operatingMargins":
                output_key = "operating_margin"

            elif field == "returnOnEquity":
                output_key = "roe"

            elif field == "returnOnAssets":
                output_key = "roa"

            elif field == "debtToEquity":
                output_key = "debt_to_equity"

            elif field == "currentRatio":
                output_key = "current_ratio"

            elif field == "revenueGrowth":
                output_key = "snapshot_revenue_growth"

            elif field == "earningsGrowth":
                output_key = "snapshot_earnings_growth"

            elif field == "dividendYield":
                output_key = "dividend_yield"

            else:
                continue

            result[output_key] = safe_float(
                info.get(key)
            )

    except Exception as e:

        print(f"  Snapshot warning: {e}")

    # --------------------------------------------------
    # INCOME STATEMENT
    # --------------------------------------------------

    try:

        income = ticker.income_stmt

        if income is not None and not income.empty:

            result["financial_years"] = len(income.columns)

            revenues = []
            net_income = []
            eps = []
            operating_income = []

            for col in income.columns:

                revenues.append(
                    get_statement_value(
                        income,
                        "Total Revenue",
                        col
                    )
                )

                net_income.append(
                    get_statement_value(
                        income,
                        "Net Income",
                        col
                    )
                )

                eps_value = get_statement_value(
                    income,
                    "Diluted EPS",
                    col
                )

                if not np.isfinite(eps_value):

                    eps_value = get_statement_value(
                        income,
                        "Basic EPS",
                        col
                    )

                eps.append(eps_value)

                operating_income.append(
                    get_statement_value(
                        income,
                        "Operating Income",
                        col
                    )
                )

            result["revenue_cagr"] = calculate_growth(
                revenues
            )

            result["net_income_cagr"] = calculate_growth(
                net_income
            )

            result["eps_cagr"] = calculate_growth(
                eps
            )

            margins = []

            for revenue, op_income in zip(
                revenues,
                operating_income
            ):

                if (
                    np.isfinite(revenue)
                    and revenue != 0
                    and np.isfinite(op_income)
                ):

                    margins.append(
                        op_income / revenue * 100
                    )

                else:

                    margins.append(np.nan)

            valid_margins = [
                x for x in margins
                if np.isfinite(x)
            ]

            if valid_margins:

                result["operating_margin_latest"] = (
                    valid_margins[0]
                )

                result["operating_margin_oldest"] = (
                    valid_margins[-1]
                )

                result["operating_margin_change"] = (
                    valid_margins[0]
                    - valid_margins[-1]
                )

    except Exception as e:

        print(f"  Income statement warning: {e}")

    # --------------------------------------------------
    # BALANCE SHEET
    # --------------------------------------------------

    try:

        balance = ticker.balance_sheet

        if balance is not None and not balance.empty:

            latest_col = balance.columns[0]

            equity = get_statement_value(
                balance,
                "Stockholders Equity",
                latest_col
            )

            if not np.isfinite(equity):

                equity = get_statement_value(
                    balance,
                    "Common Stock Equity",
                    latest_col
                )

            assets = get_statement_value(
                balance,
                "Total Assets",
                latest_col
            )

            debt = get_statement_value(
                balance,
                "Total Debt",
                latest_col
            )

            current_assets = get_statement_value(
                balance,
                "Current Assets",
                latest_col
            )

            current_liabilities = get_statement_value(
                balance,
                "Current Liabilities",
                latest_col
            )

            # Calculated ROE / ROA
            try:

                income = ticker.income_stmt

                net_income = get_statement_value(
                    income,
                    "Net Income",
                    income.columns[0]
                )

                if (
                    np.isfinite(net_income)
                    and np.isfinite(equity)
                    and equity != 0
                ):

                    result["calculated_roe"] = (
                        net_income / equity * 100
                    )

                if (
                    np.isfinite(net_income)
                    and np.isfinite(assets)
                    and assets != 0
                ):

                    result["calculated_roa"] = (
                        net_income / assets * 100
                    )

            except Exception:
                pass

            # Debt / equity
            if (
                np.isfinite(debt)
                and np.isfinite(equity)
                and equity != 0
            ):

                result["calculated_debt_to_equity"] = (
                    debt / equity * 100
                )

            # Current ratio
            if (
                np.isfinite(current_assets)
                and np.isfinite(current_liabilities)
                and current_liabilities != 0
            ):

                result["calculated_current_ratio"] = (
                    current_assets
                    / current_liabilities
                )

    except Exception as e:

        print(f"  Balance sheet warning: {e}")

    # --------------------------------------------------
    # DATA QUALITY
    # --------------------------------------------------

    important = [
        result["revenue_cagr"],
        result["net_income_cagr"],
        result["eps_cagr"],
        result["operating_margin_latest"],
    ]

    available = sum(
        np.isfinite(x)
        for x in important
    )

    if available >= 4:
        result["fundamental_data_quality"] = "HIGH"

    elif available >= 2:
        result["fundamental_data_quality"] = "MEDIUM"

    elif available >= 1:
        result["fundamental_data_quality"] = "LOW"

    else:
        result["fundamental_data_quality"] = "NONE"

    return result


def main():

    print("=" * 70)
    print("FUNDAMENTALS ENGINE")
    print("=" * 70)

    universe = pd.read_csv(
        UNIVERSE_FILE
    )

    universe = universe[
        universe["tradable"] == True
    ].copy()

    print()
    print("Stocks:", len(universe))

    os.makedirs(
        RAW_DIR,
        exist_ok=True
    )

    results = []

    for _, stock in universe.iterrows():

        try:

            result = process_stock(stock)

            results.append(result)

        except Exception as e:

            print(
                f"ERROR {stock['nse_symbol']}: {e}"
            )

        time.sleep(0.2)

    df = pd.DataFrame(results)

    # --------------------------------------------------
    # FUNDAMENTAL QUALITY SCORE
    # --------------------------------------------------
    #
    # This is deliberately NOT the final investment score.
    #
    # It rewards:
    # - positive revenue growth
    # - positive earnings growth
    # - positive EPS growth
    # - profitability
    # - improving operating margins
    #
    # Valuation is kept separate for now because
    # valuation must be interpreted relative to sector.
    # --------------------------------------------------

    score = pd.Series(
        50.0,
        index=df.index
    )

    score += np.where(
        df["revenue_cagr"] > 10,
        10,
        np.where(
            df["revenue_cagr"] > 0,
            5,
            np.where(
                df["revenue_cagr"] < 0,
                -5,
                0
            )
        )
    )

    score += np.where(
        df["net_income_cagr"] > 10,
        10,
        np.where(
            df["net_income_cagr"] > 0,
            5,
            np.where(
                df["net_income_cagr"] < 0,
                -5,
                0
            )
        )
    )

    score += np.where(
        df["eps_cagr"] > 10,
        10,
        np.where(
            df["eps_cagr"] > 0,
            5,
            np.where(
                df["eps_cagr"] < 0,
                -5,
                0
            )
        )
    )

    score += np.where(
        df["operating_margin_change"] > 2,
        10,
        np.where(
            df["operating_margin_change"] > 0,
            5,
            np.where(
                df["operating_margin_change"] < -2,
                -5,
                0
            )
        )
    )

    score += np.where(
        df["calculated_roe"] > 15,
        10,
        np.where(
            df["calculated_roe"] > 8,
            5,
            0
        )
    )

    df["fundamental_quality_score"] = (
        score.clip(0, 100)
        .round(2)
    )

    # --------------------------------------------------
    # SAVE
    # --------------------------------------------------

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("=" * 70)
    print("FUNDAMENTALS COMPLETE")
    print("=" * 70)

    print()
    print("Stocks processed:", len(df))

    print()
    print("DATA QUALITY")
    print(
        df["fundamental_data_quality"]
        .value_counts()
        .to_string()
    )

    print()
    print("TOP FUNDAMENTAL QUALITY")
    print("-" * 70)

    columns = [
        "symbol",
        "company_name",
        "sector",
        "fundamental_quality_score",
        "revenue_cagr",
        "net_income_cagr",
        "eps_cagr",
        "calculated_roe",
        "operating_margin_change",
        "fundamental_data_quality",
    ]

    print(
        df.sort_values(
            "fundamental_quality_score",
            ascending=False
        )[columns]
        .head(30)
        .to_string(index=False)
    )

    print()
    print("Saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
