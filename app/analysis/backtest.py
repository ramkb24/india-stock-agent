from pathlib import Path
import numpy as np
import pandas as pd


PRICE_DIR = Path("data/prices")
OUTPUT_DIR = Path("data/market")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "2022-01-03"
END_DATE = "2026-09-25"

REBALANCE_DAYS = 20

HORIZONS = {
    "5D": 5,
    "20D": 20,
    "60D": 60,
    "120D": 120,
}


def load_prices():
    """
    Load all historical stock prices into one dictionary.
    """

    prices = {}

    files = sorted(PRICE_DIR.glob("*.parquet"))

    print(f"Loading {len(files)} price files...")

    for i, path in enumerate(files, 1):
        try:
            df = pd.read_parquet(path)

            if df.empty:
                continue

            df.index = pd.to_datetime(df.index)

            # Remove timezone for easier alignment
            if getattr(df.index, "tz", None) is not None:
                df.index = df.index.tz_localize(None)

            df = df.sort_index()

            prices[path.stem] = df

        except Exception as exc:
            print(f"Failed: {path.name}: {exc}")

        if i % 100 == 0:
            print(f"Loaded {i}/{len(files)}")

    print(f"Successfully loaded: {len(prices)}")

    return prices


def calculate_signal_score(df):
    """
    Historical version of the LIVE technical scoring engine.

    Every indicator is calculated using data available on or
    before the signal date. No future data is used.
    """

    close = pd.to_numeric(df["Close"], errors="coerce")
    high = pd.to_numeric(df["High"], errors="coerce")
    low = pd.to_numeric(df["Low"], errors="coerce")
    volume = pd.to_numeric(df["Volume"], errors="coerce")

    # --------------------------------------------------------
    # MOVING AVERAGES
    # --------------------------------------------------------

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma100 = close.rolling(100).mean()
    sma200 = close.rolling(200).mean()

    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + rs))

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    ema12 = close.ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = close.ewm(
        span=26,
        adjust=False
    ).mean()

    macd = ema12 - ema26

    macd_signal = macd.ewm(
        span=9,
        adjust=False
    ).mean()

    macd_hist = macd - macd_signal

    # --------------------------------------------------------
    # ATR / ADX
    # --------------------------------------------------------

    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    atr14 = true_range.rolling(14).mean()

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where(
            (up_move > down_move) & (up_move > 0),
            up_move,
            0.0
        ),
        index=df.index
    )

    minus_dm = pd.Series(
        np.where(
            (down_move > up_move) & (down_move > 0),
            down_move,
            0.0
        ),
        index=df.index
    )

    tr14 = true_range.rolling(14).sum()

    plus_di = (
        100 * plus_dm.rolling(14).sum()
        / tr14.replace(0, np.nan)
    )

    minus_di = (
        100 * minus_dm.rolling(14).sum()
        / tr14.replace(0, np.nan)
    )

    dx = (
        100
        * (plus_di - minus_di).abs()
        / (plus_di + minus_di).replace(0, np.nan)
    )

    adx14 = dx.rolling(14).mean()

    # --------------------------------------------------------
    # RETURNS
    # --------------------------------------------------------

    return20 = close.pct_change(20) * 100
    return60 = close.pct_change(60) * 100
    return120 = close.pct_change(120) * 100

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    volume_sma20 = volume.rolling(20).mean()

    relative_volume = (
        volume
        / volume_sma20.replace(0, np.nan)
    )

    # --------------------------------------------------------
    # 52-WEEK HIGH / LOW
    #
    # Exclude current day to avoid look-ahead.
    # --------------------------------------------------------

    previous_252_high = (
        close.shift(1)
        .rolling(252)
        .max()
    )

    previous_252_low = (
        close.shift(1)
        .rolling(252)
        .min()
    )

    # --------------------------------------------------------
    # CONDITIONS
    # --------------------------------------------------------

    above_sma20 = close > sma20
    above_sma50 = close > sma50
    above_sma100 = close > sma100
    above_sma200 = close > sma200

    bullish_alignment = (
        (sma20 > sma50)
        & (sma50 > sma100)
        & (sma100 > sma200)
    )

    bearish_alignment = (
        (sma20 < sma50)
        & (sma50 < sma100)
        & (sma100 < sma200)
    )

    macd_bullish = macd > macd_signal

    strong_trend = adx14 >= 25

    adx_bullish = (
        (adx14 >= 20)
        & (plus_di > minus_di)
    )

    breakout_52w = (
        close > previous_252_high
    )

    breakdown_52w = (
        close < previous_252_low
    )

    high_volume = relative_volume >= 1.25

    very_high_volume = relative_volume >= 1.50

    # --------------------------------------------------------
    # LIVE TECHNICAL SCORE
    # --------------------------------------------------------

    score = pd.Series(
        50.0,
        index=df.index
    )

    def add(condition, points):
        condition = condition.fillna(False)
        score.loc[condition] += points

    add(above_sma20, 5)
    add(above_sma50, 5)
    add(above_sma100, 5)
    add(above_sma200, 5)

    add(bullish_alignment, 8)
    add(bearish_alignment, -8)

    add(macd_bullish, 5)
    add(strong_trend, 4)
    add(adx_bullish, 4)

    add(breakout_52w, 8)
    add(breakdown_52w, -8)

    add(high_volume, 2)
    add(very_high_volume, 3)

    # Momentum
    score += (
        return20.clip(-20, 20)
        * 0.25
    )

    score += (
        return60.clip(-30, 30)
        * 0.15
    )

    score += (
        return120.clip(-50, 50)
        * 0.10
    )

    # RSI
    score.loc[
        (rsi >= 50) & (rsi <= 70)
    ] += 5

    score.loc[
        rsi < 30
    ] -= 3

    score.loc[
        rsi > 80
    ] -= 5

    return score.clip(0, 100)

def add_forward_returns(df):
    """
    Calculate future returns.

    These are deliberately shifted forward and are NEVER
    used in the signal calculation.
    """

    for name, days in HORIZONS.items():
        df[f"FORWARD_{name}"] = (
            df["Close"].shift(-days) / df["Close"] - 1
        )

    return df


def build_observations(prices):
    """
    Create historical observations from all stocks.
    """

    observations = []

    for counter, (symbol, df) in enumerate(prices.items(), 1):

        if len(df) < 250:
            continue

        work = df.copy()

        work["SIGNAL_SCORE"] = calculate_signal_score(work)

        work = add_forward_returns(work)

        work = work.loc[
            START_DATE:END_DATE
        ]

        work["SYMBOL"] = symbol

        columns = [
            "SYMBOL",
            "SIGNAL_SCORE",
            "Close",
            "FORWARD_5D",
            "FORWARD_20D",
            "FORWARD_60D",
            "FORWARD_120D",
        ]

        work = work[columns]

        work = work.dropna(
            subset=["SIGNAL_SCORE"]
        )

        observations.append(work.reset_index())

        if counter % 100 == 0:
            print(
                f"Processed {counter}/{len(prices)}"
            )

    result = pd.concat(
        observations,
        ignore_index=True
    )

    result.rename(
        columns={"Date": "SIGNAL_DATE"},
        inplace=True
    )

    return result


def summarize_buckets(observations):

    observations = observations.copy()

    observations["SCORE_BUCKET"] = pd.cut(
        observations["SIGNAL_SCORE"],
        bins=[
            -0.01,
            40,
            50,
            60,
            70,
            80,
            90,
            100.01,
        ],
        labels=[
            "<40",
            "40-50",
            "50-60",
            "60-70",
            "70-80",
            "80-90",
            "90-100",
        ],
    )

    rows = []

    for bucket, group in observations.groupby(
        "SCORE_BUCKET",
        observed=False
    ):

        row = {
            "score_bucket": str(bucket),
            "observations": len(group),
            "stocks": group["SYMBOL"].nunique(),
        }

        for horizon in HORIZONS:

            col = f"FORWARD_{horizon}"

            row[f"{horizon}_mean"] = (
                group[col].mean() * 100
            )

            row[f"{horizon}_median"] = (
                group[col].median() * 100
            )

            row[f"{horizon}_positive_pct"] = (
                (group[col] > 0).mean() * 100
            )

        rows.append(row)

    return pd.DataFrame(rows)


def main():

    print("=" * 80)
    print("HISTORICAL FACTOR BACKTEST")
    print("=" * 80)

    print("\nLoading historical prices...")

    prices = load_prices()

    print("\nBuilding historical observations...")

    observations = build_observations(prices)

    print("\nObservations:", len(observations))

    print(
        "Date range:",
        observations["SIGNAL_DATE"].min(),
        "->",
        observations["SIGNAL_DATE"].max(),
    )

    print(
        "Unique stocks:",
        observations["SYMBOL"].nunique(),
    )

    print("\nGenerating score buckets...")

    summary = summarize_buckets(observations)

    print("\n")
    print("=" * 80)
    print("SCORE BUCKET PERFORMANCE")
    print("=" * 80)

    print(
        summary.to_string(index=False)
    )

    observations_path = (
        OUTPUT_DIR /
        "historical_backtest_observations.csv"
    )

    summary_path = (
        OUTPUT_DIR /
        "historical_backtest_summary.csv"
    )

    observations.to_csv(
        observations_path,
        index=False
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    print("\nSaved:")
    print(observations_path)
    print(summary_path)


if __name__ == "__main__":
    main()
