import numpy as np
import pandas as pd

from config import TRADING_DAYS, TAIL_LEVEL, NEWEY_WEST_LAGS

LADDER = {
    "CAPM": ["Mkt-RF"],
    "FF3": ["Mkt-RF", "SMB", "HML"],
    "FF5": ["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
    "FF5 + Mom": ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"],
}


def max_drawdown(r):
    wealth = (1 + r).cumprod()
    return (wealth / wealth.cummax() - 1).min()


def tail(r, rf):
    ex = r - rf
    cut = r.quantile(TAIL_LEVEL)
    lpsd = np.sqrt(np.mean(np.minimum(ex, 0.0) ** 2))
    return -cut, -r[r <= cut].mean(), ex.mean() / lpsd * np.sqrt(TRADING_DAYS)


def summary(r, rf, turnover=None, bench=None):
    rf = rf.reindex(r.index).fillna(0.0)
    ex = r - rf
    years = len(r) / TRADING_DAYS
    sharpe = ex.mean() / ex.std(ddof=1) * np.sqrt(TRADING_DAYS)
    var, es, sortino = tail(r, rf)
    out = {
        "Annualized return": (1 + r).prod() ** (1 / years) - 1,
        "Annualized volatility": r.std(ddof=1) * np.sqrt(TRADING_DAYS),
        "Sharpe ratio": sharpe,
        "t-stat of Sharpe": sharpe * np.sqrt(years),
        "Sortino ratio": sortino,
        "Daily VaR 95%": var,
        "Daily ES 95%": es,
        "Maximum drawdown": max_drawdown(r),
        "Annualized turnover": np.nan if turnover is None else turnover.mean() * 12,
    }
    if bench is not None:
        active = (r - bench.reindex(r.index)).dropna()
        te = active.std(ddof=1) * np.sqrt(TRADING_DAYS)
        ir = active.mean() * TRADING_DAYS / te
        out.update({"Tracking error": te, "Information ratio": ir,
                    "t-stat of IR": ir * np.sqrt(years)})
    return out


def newey_west_ols(y, X, lags=NEWEY_WEST_LAGS):
    X = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    xe = X * e[:, None]
    S = xe.T @ xe
    for lag in range(1, lags + 1):
        g = xe[lag:].T @ xe[:-lag]
        S += (1 - lag / (lags + 1)) * (g + g.T)
    XtX_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(XtX_inv @ S @ XtX_inv))
    r2 = 1 - e.var() / y.var()
    return beta, beta / se, r2


def regress(r, factors, names):
    data = pd.concat([r.rename("r"), factors], axis=1, join="inner").dropna()
    y = (data["r"] - data["RF"]).values
    beta, t, r2 = newey_west_ols(y, data[names].values)
    return beta, t, r2


def ladder(r, factors):
    rows = {}
    for model, names in LADDER.items():
        beta, t, r2 = regress(r, factors, names)
        rows[model] = {"Alpha (annual)": beta[0] * TRADING_DAYS, "t-stat": t[0], "R-squared": r2}
    return pd.DataFrame(rows).T


def loadings(r, factors):
    names = LADDER["FF5 + Mom"]
    beta, t, r2 = regress(r, factors, names)
    coef = pd.Series(beta, index=["Alpha (annual)"] + names)
    coef["Alpha (annual)"] *= TRADING_DAYS
    tstat = pd.Series(t, index=["Alpha (annual)"] + names)
    return coef, tstat, r2
