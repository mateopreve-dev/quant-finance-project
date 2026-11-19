import numpy as np
import pandas as pd

from config import (
    MEMBERSHIP_FILE, GICS_CURRENT_FILE, GICS_FORMER_FILE, PRICES_FILE, CLOSE_FILE,
    SHARES_FILE, FACTORS_FILE, AQR_FILE, MISSING_FILE, GROUPS, RISK_FREE, MARKET, COPPER,
    PRICE_START, PRICE_END, FIRST_TRADE, LAST_DAY, SHARES_LAG_DAYS, AQR_URL, AQR_FILES,
)


def membership_table():
    table = pd.read_csv(MEMBERSHIP_FILE, parse_dates=["date"]).set_index("date").sort_index()
    return table["tickers"].str.split(",")


def candidates():
    gics = pd.concat([pd.read_csv(GICS_CURRENT_FILE), pd.read_csv(GICS_FORMER_FILE)])
    gics = gics.drop_duplicates("Symbol").set_index("Symbol")
    group_of = {sub: group for group, subs in GROUPS.items() for sub in subs}
    gics["Group"] = gics["GICS Sub-Industry"].map(group_of)
    members = membership_table().loc[pd.Timestamp(FIRST_TRADE) - pd.DateOffset(years=1):LAST_DAY]
    ever = set(t for row in members for t in row)
    universe = gics[gics["Group"].notna() & gics.index.isin(ever)]
    return universe[["Security", "GICS Sub-Industry", "Group"]].sort_index()


def membership_matrix(dates, tickers):
    table = membership_table()
    rows = table.reindex(pd.DatetimeIndex(dates), method="ffill")
    return pd.DataFrame({t: rows.map(lambda s: t in s) for t in tickers}, index=dates)


def _fetch(symbols):
    import yfinance as yf

    raw = yf.download(symbols, start=PRICE_START, end=PRICE_END, auto_adjust=False,
                      progress=False, threads=False)
    out = []
    for field in ["Adj Close", "Close"]:
        frame = raw[field]
        if isinstance(frame, pd.Series):
            frame = frame.to_frame(symbols[0])
        out.append(frame.reindex(columns=symbols))
    return out


def download_prices(tickers):
    import tempfile
    import yfinance as yf

    yf.set_tz_cache_location(tempfile.mkdtemp())
    symbols = list(tickers) + [RISK_FREE, MARKET, COPPER]
    total, close = _fetch(symbols)
    for _ in range(3):
        for s in [s for s in symbols if total[s].notna().sum() < 50]:
            t, c = _fetch([s])
            total = total.combine_first(t)[symbols]
            close = close.combine_first(c)[symbols]
    missing = [s for s in symbols if total[s].notna().sum() < 50]
    total = total.drop(columns=missing).dropna(how="all")
    close = close.drop(columns=missing).loc[total.index]
    total.to_csv(PRICES_FILE)
    close.to_csv(CLOSE_FILE)
    pd.Series(missing, name="ticker").to_csv(MISSING_FILE, index=False)
    return total, close


def _naive(index):
    index = pd.DatetimeIndex(index)
    return (index.tz_convert(None) if index.tz is not None else index).normalize()


def _clean_shares(raw, splits):
    s = pd.Series(raw, dtype=float).dropna()
    s = s[s > 0]
    s.index = _naive(s.index)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    if s.empty:
        return s
    if splits is not None and len(splits):
        splits = pd.Series(splits, dtype=float)
        splits.index = _naive(splits.index)
        for date, ratio in splits.items():
            if not ratio or ratio == 1:
                continue
            before = s[(s.index < date) & (s.index >= date - pd.Timedelta(days=200))]
            after = s[(s.index >= date) & (s.index <= date + pd.Timedelta(days=200))]
            if before.empty or after.empty:
                continue
            jump = after.median() / before.median()
            if abs(np.log(jump / ratio)) < abs(np.log(jump)):
                s[s.index < date] *= ratio
    s = s.resample("ME").median().dropna()
    ref = s.rolling(5, center=True, min_periods=1).median()
    return s[(s / ref).between(0.67, 1.5)]


def download_shares(tickers, dates):
    import yfinance as yf

    dates = pd.DatetimeIndex(dates)
    panel, report = {}, []
    for t in tickers:
        tk = yf.Ticker(t)
        info = tk.get_info() or {}
        snapshot = info.get("sharesOutstanding")
        implied = info.get("impliedSharesOutstanding")
        try:
            raw = tk.get_shares_full(start=PRICE_START, end=PRICE_END)
        except Exception:
            raw = None
        series = _clean_shares(raw if raw is not None else [], tk.splits)
        source = "history"
        if series.empty:
            series = pd.Series(float(snapshot), index=[dates[0]])
            source = "snapshot"
        scale = 1.0
        if snapshot and implied and 1.05 < implied / snapshot < 3.0 \
                and abs(np.log(series.iloc[-1] / snapshot)) < 0.05:
            scale = implied / snapshot
        daily = series.reindex(series.index.union(dates)).ffill().reindex(dates)
        panel[t] = daily.shift(SHARES_LAG_DAYS).bfill() * scale
        report.append({"ticker": t, "source": source, "class_scale": round(scale, 4)})
    panel = pd.DataFrame(panel)
    panel.to_csv(SHARES_FILE)
    pd.DataFrame(report).to_csv(SHARES_FILE.with_name("shares_outstanding_report.csv"), index=False)
    return panel


def download_factors():
    import pandas_datareader.data as web

    ff5 = web.DataReader("F-F_Research_Data_5_Factors_2x3_daily", "famafrench",
                         start=FIRST_TRADE, end=PRICE_END)[0]
    mom = web.DataReader("F-F_Momentum_Factor_daily", "famafrench",
                         start=FIRST_TRADE, end=PRICE_END)[0]
    for frame in (ff5, mom):
        frame.index = pd.to_datetime(frame.index.astype(str))
        frame.columns = [c.strip() for c in frame.columns]
    factors = ff5.join(mom, how="inner") / 100.0
    factors.to_csv(FACTORS_FILE)
    return factors


def _aqr_factor(name, file):
    import io
    import requests

    raw = requests.get(AQR_URL.format(file), headers={"User-Agent": "Mozilla/5.0"}, timeout=600)
    raw.raise_for_status()
    sheets = pd.read_excel(io.BytesIO(raw.content), sheet_name=None, header=None)
    table = sheets.get(f"{name} Factors", next(iter(sheets.values())))
    top = table.index[table[0].astype(str).str.strip() == "DATE"][0]
    data = table.iloc[top + 1:].set_axis(table.iloc[top].astype(str).str.strip(), axis=1)
    series = pd.to_numeric(data.set_index("DATE")["USA"], errors="coerce")
    series.index = pd.to_datetime(series.index)
    return series.loc[FIRST_TRADE:LAST_DAY].dropna().rename(name)


def download_aqr():
    aqr = pd.concat([_aqr_factor(k, v) for k, v in AQR_FILES.items()], axis=1)
    aqr.to_csv(AQR_FILE)
    return aqr


def download_all():
    tickers = candidates().index.tolist()
    total, _ = download_prices(tickers)
    have = [t for t in tickers if t in total.columns]
    download_shares(have, total.index)
    download_factors()
    download_aqr()


def load():
    if not all(f.exists() for f in [PRICES_FILE, CLOSE_FILE, SHARES_FILE, FACTORS_FILE, AQR_FILE]):
        download_all()
    total = pd.read_csv(PRICES_FILE, index_col=0, parse_dates=True)
    close = pd.read_csv(CLOSE_FILE, index_col=0, parse_dates=True)
    shares = pd.read_csv(SHARES_FILE, index_col=0, parse_dates=True)
    factors = pd.read_csv(FACTORS_FILE, index_col=0, parse_dates=True)
    factors = factors.join(pd.read_csv(AQR_FILE, index_col=0, parse_dates=True))
    universe = candidates()
    universe = universe[universe.index.isin(total.columns)]
    return total, close, shares, factors, universe
