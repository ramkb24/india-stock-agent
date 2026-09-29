from pathlib import Path
import json
import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[2]

TECHNICAL_FILE = BASE / "data" / "market" / "technical_indicators.csv"
FUNDAMENTAL_FILE = BASE / "data" / "fundamentals.csv"
RISK_FILE = BASE / "data" / "market" / "stock_risk.csv"
RS_FILE = BASE / "data" / "market" / "stock_sector_relative_strength.json"
SECTOR_FILE = BASE / "data" / "market" / "sector_analysis.json"
MARKET_FILE = BASE / "data" / "market" / "market_regime.json"

OUTPUT_FILE = BASE / "data" / "market" / "stock_scores.csv"
SUMMARY_FILE = BASE / "data" / "market" / "stock_scores_summary.json"


# ============================================================
# HELPERS
# ============================================================

def clean_symbol(series):
    return (
        series.astype(str)
        .str.strip()
        .str.upper()
        .str.replace("_NS", "", regex=False)
    )


def percentile_score(series, higher_is_better=True):
    """
    Convert a numeric factor to 0-100 cross-sectional percentile.
    Missing values remain NaN.
    """
    x = pd.to_numeric(series, errors="coerce")

    if not higher_is_better:
        x = -x

    valid = x.notna()

    result = pd.Series(np.nan, index=series.index, dtype=float)

    if valid.sum() > 1:
        result.loc[valid] = x.loc[valid].rank(
            pct=True,
            method="average"
        ) * 100.0
    elif valid.sum() == 1:
        result.loc[valid] = 50.0

    return result


def weighted_average(parts):
    """
    parts = [(series, weight), ...]
    Calculate weighted average using only available factors.
    """
    numerator = pd.Series(0.0, index=parts[0][0].index)
    denominator = pd.Series(0.0, index=parts[0][0].index)

    for values, weight in parts:
        values = pd.to_numeric(values, errors="coerce")
        valid = values.notna()

        numerator.loc[valid] += values.loc[valid] * weight
        denominator.loc[valid] += weight

    result = numerator / denominator.replace(0, np.nan)

    return result


# ============================================================
# TECHNICAL SCORE
# ============================================================

def calculate_technical_score(df):

    score = pd.Series(50.0, index=df.index)

    def add(condition, points):
        nonlocal score
        condition = condition.fillna(False)
        score.loc[condition] += points

    add(df["ABOVE_SMA_20"], 5)
    add(df["ABOVE_SMA_50"], 5)
    add(df["ABOVE_SMA_100"], 5)
    add(df["ABOVE_SMA_200"], 5)

    add(df["BULLISH_ALIGNMENT"], 8)
    add(df["BEARISH_ALIGNMENT"], -8)

    add(df["MACD_BULLISH"], 5)
    add(df["STRONG_TREND"], 4)
    add(df["ADX_BULLISH"], 4)

    add(df["BREAKOUT_52W"], 8)
    add(df["BREAKDOWN_52W"], -8)

    add(df["HIGH_VOLUME"], 2)
    add(df["VERY_HIGH_VOLUME"], 3)

    returns_20 = pd.to_numeric(df["RETURN_20D"], errors="coerce")
    returns_60 = pd.to_numeric(df["RETURN_60D"], errors="coerce")
    returns_120 = pd.to_numeric(df["RETURN_120D"], errors="coerce")

    score += returns_20.clip(-20, 20) * 0.25
    score += returns_60.clip(-30, 30) * 0.15
    score += returns_120.clip(-50, 50) * 0.10

    rsi = pd.to_numeric(df["RSI_14"], errors="coerce")

    score.loc[(rsi >= 50) & (rsi <= 70)] += 5
    score.loc[rsi < 30] -= 3
    score.loc[rsi > 80] -= 5

    return score.clip(0, 100)


# ============================================================
# FUNDAMENTAL SCORE
# ============================================================

def calculate_fundamental_score(df):

    # Valuation
    valuation = weighted_average([
        (percentile_score(df["trailing_pe"], False), 0.30),
        (percentile_score(df["forward_pe"], False), 0.25),
        (percentile_score(df["price_to_book"], False), 0.20),
        (percentile_score(df["enterprise_to_ebitda"], False), 0.25),
    ])

    # Growth
    growth = weighted_average([
        (percentile_score(df["revenue_cagr"], True), 0.30),
        (percentile_score(df["net_income_cagr"], True), 0.35),
        (percentile_score(df["eps_cagr"], True), 0.35),
    ])

    # Profitability
    profitability = weighted_average([
        (percentile_score(df["roe"], True), 0.30),
        (percentile_score(df["roa"], True), 0.20),
        (percentile_score(df["operating_margin"], True), 0.25),
        (percentile_score(df["operating_margin_change"], True), 0.25),
    ])

    # Balance sheet
    balance = weighted_average([
        (percentile_score(df["debt_to_equity"], False), 0.60),
        (percentile_score(df["current_ratio"], True), 0.40),
    ])

    final = weighted_average([
        (valuation, 0.30),
        (growth, 0.25),
        (profitability, 0.25),
        (balance, 0.20),
    ])

    # --------------------------------------------------------
    # FUNDAMENTAL DATA COVERAGE
    # --------------------------------------------------------
    # Measure how much underlying fundamental information exists.
    # Do not convert completely missing fundamentals into a fake
    # neutral score of 50.
    fundamental_fields = [
        "trailing_pe",
        "forward_pe",
        "price_to_book",
        "enterprise_to_ebitda",
        "revenue_cagr",
        "net_income_cagr",
        "eps_cagr",
        "roe",
        "roa",
        "operating_margin",
        "operating_margin_change",
        "debt_to_equity",
        "current_ratio",
    ]

    available_fields = [
        c for c in fundamental_fields
        if c in df.columns
    ]

    if available_fields:
        df["fundamental_data_coverage"] = (
            df[available_fields].notna().sum(axis=1)
            / len(available_fields)
        )
    else:
        df["fundamental_data_coverage"] = 0.0

    # Keep NaN when there is no usable fundamental information.
    # Partial information is already handled by weighted_average().
    return final.clip(0, 100)


# ============================================================
# MACRO SCORE
# ============================================================

def calculate_macro_score():

    try:
        path = BASE / "data" / "market" / "macro_analysis.json"

        with open(path) as f:
            macro = json.load(f)

        risk_points = float(macro.get("macro_risk_points", 0))

        score = 50.0

        # Generic market-wide macro adjustment.
        score -= risk_points * 5

        return float(np.clip(score, 0, 100))

    except Exception:
        return 50.0


# ============================================================
# RISK SCORE
# ============================================================

def calculate_risk_score(df):

    if "risk_points" in df.columns:
        points = pd.to_numeric(
            df["risk_points"],
            errors="coerce"
        ).fillna(10)

        return (100 - points * 5).clip(0, 100)

    return pd.Series(50.0, index=df.index)


# ============================================================
# LOAD ALL DATA
# ============================================================

def load_data():

    print("Loading technical data...")
    tech = pd.read_csv(TECHNICAL_FILE)

    print("Loading fundamentals...")
    fund = pd.read_csv(FUNDAMENTAL_FILE)

    print("Loading risk...")
    risk = pd.read_csv(RISK_FILE)

    print("Loading relative strength...")
    with open(RS_FILE) as f:
        rs = pd.DataFrame(json.load(f))

    print("Loading sector analysis...")
    with open(SECTOR_FILE) as f:
        sectors = pd.DataFrame(json.load(f))

    print("Loading market regime...")
    with open(MARKET_FILE) as f:
        market = json.load(f)

    # Normalize symbols
    tech["symbol_clean"] = clean_symbol(tech["symbol"])
    fund["symbol_clean"] = clean_symbol(fund["symbol"])
    risk["symbol_clean"] = clean_symbol(risk["symbol"])
    rs["symbol_clean"] = clean_symbol(rs["symbol"])

    # Remove possible duplicate symbol rows
    tech = tech.drop_duplicates("symbol_clean")
    fund = fund.drop_duplicates("symbol_clean")
    risk = risk.drop_duplicates("symbol_clean")
    rs = rs.drop_duplicates("symbol_clean")

    return tech, fund, risk, rs, sectors, market


# ============================================================
# BUILD MASTER DATASET
# ============================================================

def build_master(tech, fund, risk, rs, sectors):

    # Start with technical universe
    df = tech.copy()

    # --------------------------------------------------------
    # Fundamentals
    # --------------------------------------------------------

    fundamental_columns = [
        "symbol_clean",
        "company_name",
        "sector",
        "market_cap",
        "trailing_pe",
        "forward_pe",
        "price_to_book",
        "enterprise_to_ebitda",
        "profit_margin",
        "operating_margin",
        "roe",
        "roa",
        "debt_to_equity",
        "current_ratio",
        "snapshot_revenue_growth",
        "snapshot_earnings_growth",
        "dividend_yield",
        "revenue_cagr",
        "net_income_cagr",
        "eps_cagr",
        "operating_margin_latest",
        "operating_margin_oldest",
        "operating_margin_change",
        "fundamental_data_quality",
    ]

    fundamental_columns = [
        c for c in fundamental_columns
        if c in fund.columns
    ]

    fund_merge = fund[fundamental_columns].copy()

    df = df.merge(
        fund_merge,
        on="symbol_clean",
        how="left",
        suffixes=("", "_fund")
    )

    # --------------------------------------------------------
    # Risk
    # --------------------------------------------------------

    risk_columns = [
        "symbol_clean",
        "risk_points",
        "risk_class",
        "volatility_20d",
        "volatility_60d",
        "volatility_1y",
        "max_drawdown",
        "distance_from_52w_high",
    ]

    risk_columns = [
        c for c in risk_columns
        if c in risk.columns
    ]

    risk_merge = risk[risk_columns].copy()

    df = df.merge(
        risk_merge,
        on="symbol_clean",
        how="left",
        suffixes=("", "_risk")
    )

    # --------------------------------------------------------
    # Relative Strength
    # --------------------------------------------------------

    rs_columns = [
        "symbol_clean",
        "relative_strength_score",
        "technical_vs_sector",
        "sector",
        "sector_avg_return_20d",
        "sector_avg_return_60d",
        "sector_avg_return_120d",
        "relative_strength_20d",
        "relative_strength_60d",
        "relative_strength_120d",
        "rs_20d_percentile",
        "rs_60d_percentile",
        "rs_120d_percentile",
    ]

    rs_columns = [
        c for c in rs_columns
        if c in rs.columns
    ]

    rs_merge = rs[rs_columns].copy()

    # Do not overwrite existing sector from fundamentals
    if "sector" in rs_merge.columns and "sector" in df.columns:
        rs_merge = rs_merge.rename(columns={"sector": "rs_sector"})

    df = df.merge(
        rs_merge,
        on="symbol_clean",
        how="left",
        suffixes=("", "_rs")
    )

    # --------------------------------------------------------
    # Sector score
    # --------------------------------------------------------

    sector_map = sectors[
        ["sector", "sector_score"]
    ].copy()

    sector_map["sector"] = (
        sector_map["sector"]
        .astype(str)
        .str.strip()
    )

    df["sector_score_final"] = df["sector"].map(
        sector_map.set_index("sector")["sector_score"]
    )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    print("\nMERGE COVERAGE")
    print(
        "Fundamentals:",
        df["trailing_pe"].notna().sum(),
        "/",
        len(df)
    )
    print(
        "Relative strength:",
        df["relative_strength_score"].notna().sum(),
        "/",
        len(df)
    )
    print(
        "Sector score:",
        df["sector_score_final"].notna().sum(),
        "/",
        len(df)
    )
    print(
        "Risk:",
        df["risk_points"].notna().sum(),
        "/",
        len(df)
    )

    return df


# ============================================================
# FINAL SCORING
# ============================================================

def calculate_final_scores(df, market):

    print("\nCalculating technical scores...")
    df["technical_score_final"] = calculate_technical_score(df)

    print("Calculating fundamental scores...")
    df["fundamental_score_final"] = calculate_fundamental_score(df)

    # Relative strength
    df["relative_strength_score"] = pd.to_numeric(
        df["relative_strength_score"],
        errors="coerce"
    )

    # Sector
    df["sector_score_final"] = pd.to_numeric(
        df["sector_score_final"],
        errors="coerce"
    )

    # Macro
    macro_score = calculate_macro_score()

    df["macro_score_final"] = macro_score

    # Risk
    print("Calculating risk scores...")
    df["risk_score_final"] = calculate_risk_score(df)

    # Market regime
    regime = str(
        market.get("regime", "UNKNOWN")
    ).upper()

    regime_score = float(
        market.get("regime_score", 50)
    )

    print("\nMarket regime:", regime)
    print("Market score :", regime_score)

    df["market_regime_score"] = regime_score

    # --------------------------------------------------------
    # BASE COMPOSITE SCORE
    # Availability-aware weighted composite.
    #
    # If a factor is unavailable, its weight is redistributed
    # across the factors that are actually available.
    # --------------------------------------------------------

    factor_weights = {
        "technical_score_final": 0.25,
        "fundamental_score_final": 0.25,
        "relative_strength_score": 0.15,
        "sector_score_final": 0.10,
        "macro_score_final": 0.10,
        "risk_score_final": 0.10,
        "market_regime_score": 0.05,
    }

    weighted_sum = pd.Series(0.0, index=df.index)
    available_weight = pd.Series(0.0, index=df.index)

    for column, weight in factor_weights.items():
        if column not in df.columns:
            continue

        values = pd.to_numeric(df[column], errors="coerce")
        valid = values.notna()

        weighted_sum.loc[valid] += values.loc[valid] * weight
        available_weight.loc[valid] += weight

    df["available_factor_weight"] = available_weight

    df["base_score"] = (
        weighted_sum / available_weight.replace(0, np.nan)
    ).clip(0, 100)

    # Coverage of the original seven-factor model.
    total_factor_weight = sum(factor_weights.values())

    df["composite_data_coverage"] = (
        available_weight / total_factor_weight
    )

    # Market-regime adjustment
    regime_adjustment = {
        "BULLISH": 2.5,
        "CAUTIOUS": -2.5,
        "BEARISH": -5.0,
    }.get(regime, 0.0)

    df["regime_adjustment"] = regime_adjustment

    # Risk adjustment
    df["risk_adjustment"] = np.select(
        [
            df["risk_class"].eq("VERY_HIGH"),
            df["risk_class"].eq("HIGH"),
        ],
        [
            -6.0,
            -3.0,
        ],
        default=0.0,
    )

    df["final_score"] = (
        df["base_score"]
        + df["regime_adjustment"]
        + df["risk_adjustment"]
    ).clip(0, 100)


    # --------------------------------------------------------
    # Missing-factor handling
    # --------------------------------------------------------

    # --------------------------------------------------------
    # DATA CONFIDENCE
    # Measure underlying data availability, not final scores.
    # --------------------------------------------------------

    fundamental_fields = [
        "trailing_pe",
        "forward_pe",
        "price_to_book",
        "enterprise_to_ebitda",
        "revenue_cagr",
        "net_income_cagr",
        "eps_cagr",
        "roe",
        "roa",
        "operating_margin",
        "operating_margin_change",
        "debt_to_equity",
        "current_ratio",
    ]

    fundamental_available = df[
        [c for c in fundamental_fields if c in df.columns]
    ].notna().sum(axis=1)

    # Technical data is considered available when the core price/
    # indicator history required by the technical engine exists.
    technical_fields = [
        "SMA_20",
        "SMA_50",
        "SMA_200",
        "RSI_14",
        "MACD",
        "ATR_14",
        "RETURN_20D",
        "RETURN_60D",
    ]

    technical_available = df[
        [c for c in technical_fields if c in df.columns]
    ].notna().sum(axis=1)

    # Relative strength coverage
    rs_available = df[
        [c for c in [
            "relative_strength_score",
            "relative_strength_20d",
            "relative_strength_60d",
            "relative_strength_120d",
        ] if c in df.columns]
    ].notna().sum(axis=1)

    # Sector and risk are single-factor availability checks.
    sector_available = df["sector_score_final"].notna().astype(int)

    risk_available = df["risk_points"].notna().astype(int)

    # Convert each component to a normalized availability score.
    fundamental_coverage = (
        fundamental_available / max(len(fundamental_fields), 1)
    )

    technical_coverage = (
        technical_available / max(len(technical_fields), 1)
    )

    rs_coverage = (
        rs_available / 4.0
    )

    # Weighted data coverage.
    data_coverage = (
        fundamental_coverage * 0.35
        + technical_coverage * 0.25
        + rs_coverage * 0.15
        + sector_available * 0.10
        + risk_available * 0.15
    )

    df["fundamental_data_count"] = fundamental_available
    df["technical_data_count"] = technical_available
    df["data_coverage"] = data_coverage

    df["data_confidence"] = np.select(
        [
            data_coverage >= 0.85,
            data_coverage >= 0.70,
            data_coverage >= 0.50,
        ],
        [
            "HIGH",
            "MEDIUM",
            "LOW",
        ],
        default="VERY_LOW",
    )

    # Confidence adjustment
    confidence_adjustment = np.select(
        [
            df["data_confidence"] == "HIGH",
            df["data_confidence"] == "MEDIUM",
            df["data_confidence"] == "LOW",
        ],
        [
            0.0,
            -1.0,
            -3.0,
        ],
        default=-5.0
    )

    df["confidence_adjustment"] = confidence_adjustment

    df["final_score"] = (
        df["final_score"]
        + df["confidence_adjustment"]
    ).clip(0, 100)

    # Rank
    df = df.sort_values(
        ["final_score", "technical_score_final"],
        ascending=[False, False]
    ).reset_index(drop=True)

    df["rank"] = np.arange(1, len(df) + 1)

    # Score bands
    df["score_band"] = pd.cut(
        df["final_score"],
        bins=[-np.inf, 40, 50, 60, 70, 80, 90, np.inf],
        labels=[
            "VERY_LOW",
            "LOW",
            "NEUTRAL",
            "GOOD",
            "STRONG",
            "VERY_STRONG",
            "EXCEPTIONAL",
        ]
    )

    return df


# ============================================================
# MAIN
# ============================================================

def run():

    print("=" * 70)
    print("FINAL MULTI-FACTOR STOCK SCORING ENGINE")
    print("=" * 70)

    tech, fund, risk, rs, sectors, market = load_data()

    print("\nINPUT ROWS")
    print("Technical       :", len(tech))
    print("Fundamentals    :", len(fund))
    print("Risk            :", len(risk))
    print("Relative strength:", len(rs))

    df = build_master(
        tech,
        fund,
        risk,
        rs,
        sectors
    )

    df = calculate_final_scores(
        df,
        market
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    summary = {
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "stocks_scored": int(len(df)),
        "market_regime": market.get("regime"),
        "market_regime_score": market.get("regime_score"),
        "macro_score": calculate_macro_score(),
        "score_statistics": {
            "mean": float(df["final_score"].mean()),
            "median": float(df["final_score"].median()),
            "min": float(df["final_score"].min()),
            "max": float(df["final_score"].max()),
        },
        "confidence_distribution": (
            df["data_confidence"]
            .value_counts()
            .to_dict()
        ),
        "risk_distribution": (
            df["risk_class"]
            .value_counts()
            .to_dict()
        ),
    }

    with open(
        SUMMARY_FILE,
        "w"
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            default=str
        )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    display_columns = [
        "rank",
        "symbol",
        "company_name",
        "sector",
        "final_score",
        "technical_score_final",
        "fundamental_score_final",
        "relative_strength_score",
        "sector_score_final",
        "macro_score_final",
        "risk_score_final",
        "risk_class",
        "data_confidence",
    ]

    display_columns = [
        c for c in display_columns
        if c in df.columns
    ]

    print("\n" + "=" * 70)
    print("FINAL SCORING COMPLETE")
    print("=" * 70)

    print("Stocks scored:", len(df))
    print("Market regime:", market.get("regime"))
    print(
        "Market score :",
        market.get("regime_score")
    )

    print("\nTop 20 candidates by composite score:\n")

    print(
        df.head(20)[display_columns]
        .to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("Score distribution:")
    print(df["score_band"].value_counts().sort_index())

    print("\nConfidence distribution:")
    print(df["data_confidence"].value_counts())

    print("\n" + "=" * 70)
    print("Saved:", OUTPUT_FILE)
    print("Saved:", SUMMARY_FILE)
    print("=" * 70)


if __name__ == "__main__":
    run()
