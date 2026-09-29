from pathlib import Path
import numpy as np
import pandas as pd


PRICE_DIR = Path("data/prices")
OUTPUT_DIR = Path("data/market")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "2022-03-22"
END_DATE = "2026-09-25"

REBALANCE_DAYS = 20

PORTFOLIO_SIZES = [10, 20, 50]


def load_prices():

    prices = {}

    files = sorted(PRICE_DIR.glob("*.parquet"))

    print(f"Loading {len(files)} price files...")

    for i, path in enumerate(files, 1):

        try:
            df = pd.read_parquet(path)

            if df.empty:
                continue

            df.index = pd.to_datetime(df.index)

            if getattr(df.index, "tz", None) is not None:
                df.index = df.index.tz_localize(None)

            df = df.sort_index()

            prices[path.stem] = df

        except Exception as exc:
            print(f"Failed {path.name}: {exc}")

        if i % 100 == 0:
            print(f"Loaded {i}/{len(files)}")

    return prices


def calculate_signal_score(df):

    close = pd.to_numeric(df["Close"], errors="coerce")
    high = pd.to_numeric(df["High"], errors="coerce")
    low = pd.to_numeric(df["Low"], errors="coerce")
    volume = pd.to_numeric(df["Volume"], errors="coerce")

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma100 = close.rolling(100).mean()
    sma200 = close.rolling(200).mean()

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + rs))

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

    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where(
            (up_move > down_move) & (up_move > 0),
            up_move,
            0,
        ),
        index=df.index,
    )

    minus_dm = pd.Series(
        np.where(
            (down_move > up_move) & (down_move > 0),
            down_move,
            0,
        ),
        index=df.index,
    )

    tr14 = tr.rolling(14).sum()

    plus_di = (
        100
        * plus_dm.rolling(14).sum()
        / tr14.replace(0, np.nan)
    )

    minus_di = (
        100
        * minus_dm.rolling(14).sum()
        / tr14.replace(0, np.nan)
    )

    dx = (
        100
        * (plus_di - minus_di).abs()
        / (plus_di + minus_di).replace(0, np.nan)
    )

    adx = dx.rolling(14).mean()

    return20 = close.pct_change(20) * 100
    return60 = close.pct_change(60) * 100
    return120 = close.pct_change(120) * 100

    volume_sma20 = volume.rolling(20).mean()

    relative_volume = (
        volume / volume_sma20.replace(0, np.nan)
    )

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

    above20 = close > sma20
    above50 = close > sma50
    above100 = close > sma100
    above200 = close > sma200

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

    strong_trend = adx >= 25

    adx_bullish = (
        (adx >= 20)
        & (plus_di > minus_di)
    )

    breakout = close > previous_252_high
    breakdown = close < previous_252_low

    high_volume = relative_volume >= 1.25
    very_high_volume = relative_volume >= 1.50

    score = pd.Series(
        50.0,
        index=df.index,
    )

    def add(condition, points):

        condition = condition.fillna(False)

        score.loc[condition] += points

    add(above20, 5)
    add(above50, 5)
    add(above100, 5)
    add(above200, 5)

    add(bullish_alignment, 8)
    add(bearish_alignment, -8)

    add(macd_bullish, 5)
    add(strong_trend, 4)
    add(adx_bullish, 4)

    add(breakout, 8)
    add(breakdown, -8)

    add(high_volume, 2)
    add(very_high_volume, 3)

    score += return20.clip(-20, 20) * 0.25
    score += return60.clip(-30, 30) * 0.15
    score += return120.clip(-50, 50) * 0.10

    score.loc[
        (rsi >= 50) & (rsi <= 70)
    ] += 5

    score.loc[rsi < 30] -= 3

    score.loc[rsi > 80] -= 5

    return score.clip(0, 100)


def build_signal_matrix(prices):

    scores = {}

    print("\nCalculating historical scores...")

    for i, (symbol, df) in enumerate(
        prices.items(),
        1,
    ):

        if len(df) < 300:
            continue

        score = calculate_signal_score(df)

        score = score.loc[
            START_DATE:END_DATE
        ]

        scores[symbol] = score

        if i % 100 == 0:
            print(f"Scored {i}/{len(prices)}")

    result = pd.DataFrame(scores)

    return result


def build_close_matrix(prices):

    closes = {}

    for symbol, df in prices.items():

        close = df["Close"].copy()

        close.index = pd.to_datetime(
            close.index
        )

        if getattr(close.index, "tz", None) is not None:
            close.index = close.index.tz_localize(None)

        close = close.loc[
            START_DATE:END_DATE
        ]

        closes[symbol] = close

    return pd.DataFrame(closes)


def calculate_portfolio_returns(
    scores,
    prices,
    portfolio_size,
    reverse=False,
    transaction_cost_bps=20,
):
    """
    Realistic portfolio simulation.

    Signal:
        Calculated using the closing price on T.

    Entry:
        Next trading day's OPEN.

    Exit:
        Next rebalance date's OPEN.

    transaction_cost_bps:
        Round-trip transaction-cost assumption in basis points.
        20 bps = 0.20%.

    IMPORTANT:
        We deliberately do NOT use the signal day's close for execution.
    """

    dates = scores.index

    rebalance_dates = dates[
        ::REBALANCE_DAYS
    ]

    portfolio_returns = []
    holdings_history = []

    for i in range(
        len(rebalance_dates) - 1
    ):

        signal_date = rebalance_dates[i]
        next_signal_date = rebalance_dates[i + 1]

        # ----------------------------------------------------
        # Find the FIRST trading day AFTER the signal date.
        # This is our realistic entry price.
        # ----------------------------------------------------

        future_dates = dates[
            dates > signal_date
        ]

        if len(future_dates) == 0:
            continue

        entry_date = future_dates[0]

        # ----------------------------------------------------
        # Find the first trading day on/after the next
        # rebalance signal date.
        #
        # This becomes the exit date.
        # ----------------------------------------------------

        exit_dates = dates[
            dates >= next_signal_date
        ]

        if len(exit_dates) == 0:
            continue

        exit_date = exit_dates[0]

        # ----------------------------------------------------
        # Select stocks using ONLY information available
        # at signal_date.
        # ----------------------------------------------------

        row = scores.loc[signal_date].dropna()

        if reverse:
            selected = row.nsmallest(
                portfolio_size
            )
        else:
            selected = row.nlargest(
                portfolio_size
            )

        selected = selected.index.tolist()

        if not selected:
            continue

        # ----------------------------------------------------
        # Remove stocks that don't have valid prices.
        # ----------------------------------------------------

        valid_selected = []

        for symbol in selected:

            if symbol not in prices:
                continue

            price_df = prices[symbol]

            if (
                entry_date not in price_df.index
                or exit_date not in price_df.index
            ):
                continue

            entry_price = price_df.loc[
                entry_date,
                "Open",
            ]

            exit_price = price_df.loc[
                exit_date,
                "Open",
            ]

            if (
                pd.isna(entry_price)
                or pd.isna(exit_price)
                or entry_price <= 0
                or exit_price <= 0
            ):
                continue

            valid_selected.append(symbol)

        if not valid_selected:
            continue

        # ----------------------------------------------------
        # Calculate equal-weight returns.
        # ----------------------------------------------------

        stock_returns = []

        for symbol in valid_selected:

            price_df = prices[symbol]

            entry_price = float(
                price_df.loc[
                    entry_date,
                    "Open",
                ]
            )

            exit_price = float(
                price_df.loc[
                    exit_date,
                    "Open",
                ]
            )

            stock_return = (
                exit_price / entry_price
            ) - 1

            stock_returns.append(
                stock_return
            )

        portfolio_return = np.mean(
            stock_returns
        )

        # ----------------------------------------------------
        # Transaction costs.
        #
        # Applied as a simple round-trip cost.
        # 20 bps = 0.20%.
        # ----------------------------------------------------

        transaction_cost = (
            transaction_cost_bps / 10000
        )

        portfolio_return -= transaction_cost

        portfolio_returns.append(
            {
                "signal_date": signal_date,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "return": portfolio_return,
                "holdings": len(valid_selected),
            }
        )

        holdings_history.append(
            {
                "signal_date": signal_date,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "holdings": ",".join(
                    valid_selected
                ),
            }
        )

    return (
        pd.DataFrame(portfolio_returns),
        pd.DataFrame(holdings_history),
    )

def performance_metrics(returns):

    returns = returns.dropna()

    if len(returns) == 0:
        return {}

    equity = (
        1 + returns
    ).cumprod()

    total_return = equity.iloc[-1] - 1

    years = (
        len(returns)
        * REBALANCE_DAYS
        / 252
    )

    cagr = (
        equity.iloc[-1]
        ** (1 / years)
        - 1
    )

    annualized_volatility = (
        returns.std()
        * np.sqrt(252 / REBALANCE_DAYS)
    )

    sharpe = (
        returns.mean()
        / returns.std()
        * np.sqrt(252 / REBALANCE_DAYS)
    )

    drawdown = (
        equity
        / equity.cummax()
        - 1
    )

    max_drawdown = drawdown.min()

    win_rate = (
        returns > 0
    ).mean()

    return {
        "total_return_pct": total_return * 100,
        "cagr_pct": cagr * 100,
        "annualized_volatility_pct": annualized_volatility * 100,
        "sharpe": sharpe,
        "max_drawdown_pct": max_drawdown * 100,
        "win_rate_pct": win_rate * 100,
        "periods": len(returns),
    }


def main():

    print("=" * 80)
    print("PORTFOLIO BACKTEST")
    print("=" * 80)

    prices = load_prices()

    print("\nBuilding signal matrix...")

    scores = build_signal_matrix(prices)

    print(
        "Signal matrix:",
        scores.shape,
    )

    closes = build_close_matrix(prices)

    print(
        "Close matrix:",
        closes.shape,
    )

    results = []

    for size in PORTFOLIO_SIZES:

        print(
            f"\nTesting TOP {size}..."
        )

        returns, holdings = (
            calculate_portfolio_returns(
                scores,
                prices,
                size,
                reverse=False,
            )
        )

        metrics = performance_metrics(
            returns["return"]
        )

        metrics["strategy"] = (
            f"TOP_{size}"
        )

        results.append(metrics)

        returns.to_csv(
            OUTPUT_DIR
            / f"portfolio_top_{size}_returns.csv",
            index=False,
        )

        holdings.to_csv(
            OUTPUT_DIR
            / f"portfolio_top_{size}_holdings.csv",
            index=False,
        )

        print(metrics)

        print(
            f"\nTesting BOTTOM {size}..."
        )

        returns, holdings = (
            calculate_portfolio_returns(
                scores,
                prices,
                size,
                reverse=True,
            )
        )

        metrics = performance_metrics(
            returns["return"]
        )

        metrics["strategy"] = (
            f"BOTTOM_{size}"
        )

        results.append(metrics)

        returns.to_csv(
            OUTPUT_DIR
            / f"portfolio_bottom_{size}_returns.csv",
            index=False,
        )

        holdings.to_csv(
            OUTPUT_DIR
            / f"portfolio_bottom_{size}_holdings.csv",
            index=False,
        )

        print(metrics)

    result_df = pd.DataFrame(results)

    print("\n")
    print("=" * 80)
    print("PORTFOLIO BACKTEST RESULTS")
    print("=" * 80)

    print(
        result_df[
            [
                "strategy",
                "total_return_pct",
                "cagr_pct",
                "annualized_volatility_pct",
                "sharpe",
                "max_drawdown_pct",
                "win_rate_pct",
                "periods",
            ]
        ].to_string(index=False)
    )

    result_df.to_csv(
        OUTPUT_DIR
        / "portfolio_backtest_summary.csv",
        index=False,
    )

    print(
        "\nSaved:",
        OUTPUT_DIR
        / "portfolio_backtest_summary.csv",
    )


if __name__ == "__main__":
    main()
