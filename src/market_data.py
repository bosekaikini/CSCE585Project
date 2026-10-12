"""Reproducible Yahoo Finance ingestion and parameter estimation.

Yahoo Finance is used only to obtain observed underlying prices. Option
reference prices and training targets remain model-based (analytic, Monte
Carlo, or finite-difference), not downloaded from Yahoo Finance.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd


DEFAULT_TICKERS = [
    "SPY", "QQQ", "DIA", "IWM", "EFA", "EEM", "TLT", "GLD", "USO", "XLE",
    "XLK", "XLF", "XLI", "XLP", "XLV", "XLU", "XLY", "VNQ", "HYG", "LQD",
]


def download_adjusted_close(
    tickers: Iterable[str] = DEFAULT_TICKERS,
    period: str = "5y",
    interval: str = "1d",
) -> pd.DataFrame:
    """Download adjusted close prices from Yahoo Finance.

    The returned frame has a UTC-naive date index and one column per ticker.
    Failed or empty downloads raise an explicit error instead of silently
    producing a success-shaped dataset.
    """
    import yfinance as yf

    symbols = [ticker.upper() for ticker in tickers]
    if not symbols:
        raise ValueError("at least one ticker is required")
    raw = yf.download(
        symbols, period=period, interval=interval, auto_adjust=False,
        progress=False, group_by="column", threads=False,
    )
    if raw.empty:
        raise RuntimeError("Yahoo Finance returned no rows")
    if isinstance(raw.columns, pd.MultiIndex):
        level = "Adj Close" if "Adj Close" in raw.columns.get_level_values(0) else "Close"
        prices = raw[level].copy()
    else:
        column = "Adj Close" if "Adj Close" in raw.columns else "Close"
        prices = raw[[column]].rename(columns={column: symbols[0]})
    prices = prices.reindex(columns=symbols).dropna(how="all")
    missing = [symbol for symbol in symbols if symbol not in prices or prices[symbol].dropna().empty]
    if missing:
        raise RuntimeError(f"Yahoo Finance returned no usable prices for: {', '.join(missing)}")
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    return prices.dropna(how="any")


def estimate_market_parameters(prices: pd.DataFrame, periods_per_year: int = 252) -> pd.DataFrame:
    """Estimate latest spot and annualized log-return volatility per asset."""
    if prices.empty:
        raise ValueError("prices must not be empty")
    returns = prices.apply(
        lambda series: np.log(series.dropna() / series.dropna().shift(1)).dropna()
    )
    result = pd.DataFrame({
        "ticker": prices.columns,
        "spot": prices.iloc[-1].to_numpy(dtype=float),
        "volatility": (returns.std(ddof=1) * periods_per_year**0.5).to_numpy(dtype=float),
        "observations": prices.notna().sum().to_numpy(dtype=int),
    })
    if result["volatility"].isna().any():
        raise RuntimeError("could not estimate volatility for every ticker")
    return result


def save_market_data(
    prices: pd.DataFrame,
    parameters: pd.DataFrame,
    output_dir: str | Path = "data/market",
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prices.to_csv(output / "adjusted_close.csv", index_label="date")
    parameters.to_csv(output / "parameters.csv", index=False)
    (output / "metadata.json").write_text(json.dumps({
        "source": "Yahoo Finance via yfinance",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "first_date": str(prices.index.min().date()),
        "last_date": str(prices.index.max().date()),
        "tickers": list(prices.columns),
        "volatility_method": "standard deviation of daily log returns times sqrt(252)",
    }, indent=2) + "\n")
