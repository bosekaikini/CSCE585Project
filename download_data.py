"""Download observed underlying prices used to parameterize the experiments."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.market_data import DEFAULT_TICKERS, download_adjusted_close, estimate_market_parameters, save_market_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--period", default="5y", help="Yahoo Finance period, e.g. 2y or 5y")
    parser.add_argument("--interval", default="1d", help="Yahoo Finance interval")
    parser.add_argument("--output-dir", default="data/market")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    args = parser.parse_args()
    prices = download_adjusted_close(args.tickers, args.period, args.interval)
    parameters = estimate_market_parameters(prices)
    save_market_data(prices, parameters, args.output_dir)
    print(f"Saved {len(prices)} observations for {len(prices.columns)} assets to {Path(args.output_dir)}")
    print(parameters.to_string(index=False))


if __name__ == "__main__":
    main()
