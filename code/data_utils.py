import numpy as np
import pandas as pd
import yfinance as yf

from config import (
    TICKERS, START_DATE, END_DATE,
    BENCHMARK_TICKER, RISK_FREE_TICKER,
    PRICES_CACHE, SHARES_CACHE, FACTORS_CACHE,
)


def download_prices(force=False):
    if PRICES_CACHE.exists() and not force:
        return pd.read_csv(PRICES_CACHE, index_col=0, parse_dates=True)

    all_tickers = TICKERS + [BENCHMARK_TICKER, RISK_FREE_TICKER]
    raw = yf.download(all_tickers, start=START_DATE, end=END_DATE,
                      auto_adjust=True, progress=False)
    prices = raw["Close"].copy()

    dead = prices.columns[prices.isna().all()].tolist()
    if dead:
        print(f"Warning: no data returned for {dead}")

    prices.to_csv(PRICES_CACHE)
    return prices


def download_shares_outstanding(force=False):
    if SHARES_CACHE.exists() and not force:
        return pd.read_csv(SHARES_CACHE, index_col=0).squeeze("columns")

    shares = {}
    for t in TICKERS:
        info = yf.Ticker(t).get_info()
        shares[t] = info.get("sharesOutstanding", np.nan)

    shares = pd.Series(shares, name="shares_outstanding")
    shares.to_csv(SHARES_CACHE)
    return shares


def download_factors(force=False):
    if FACTORS_CACHE.exists() and not force:
        return pd.read_csv(FACTORS_CACHE, index_col=0, parse_dates=True)

    import pandas_datareader.data as web

    ff5 = web.DataReader("F-F_Research_Data_5_Factors_2x3_daily", "famafrench",
                         start=START_DATE, end=END_DATE)[0]
    mom = web.DataReader("F-F_Momentum_Factor_daily", "famafrench",
                         start=START_DATE, end=END_DATE)[0]

    ff5.index = pd.to_datetime(ff5.index.astype(str))
    mom.index = pd.to_datetime(mom.index.astype(str))
    ff5.columns = [c.strip() for c in ff5.columns]
    mom.columns = [c.strip() for c in mom.columns]

    factors = ff5.join(mom, how="inner") / 100.0
    factors.to_csv(FACTORS_CACHE)
    return factors


def compute_returns(prices):
    return prices.pct_change().dropna(how="all")
