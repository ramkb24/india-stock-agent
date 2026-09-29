# Technical-Only Backtest Baseline v1

Created: 2026-09-29T14:16:20.501246+00:00

Git commit: `4d72564d47c3cbe57871fe02013165a3aa0f60d9`

## Purpose

This directory freezes the current technical-only research state before
introducing point-in-time universe logic or more realistic transaction costs.

## Important limitations

- Current NIFTY 500/Sensex universe is used historically.
- Therefore survivorship bias remains.
- Current fundamentals are NOT used as historical signals.
- Rebalance schedule currently uses `dates[::REBALANCE_DAYS]`.
- Transaction cost is currently a flat round-trip assumption.
- No liquidity-dependent slippage model exists yet.

## Execution

Signal is calculated at the close of T.

Entry occurs at the first available trading day's OPEN after T.

Exit occurs at the first available trading day's OPEN on/after the
next rebalance signal date.

## DO NOT MODIFY

This directory is the comparison baseline for subsequent research versions.
