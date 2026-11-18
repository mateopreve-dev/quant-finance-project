import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import ledoit_wolf

from config import (
    TRADING_DAYS, LOOKBACK, SKIP, TOP_FRACTION, MIN_HOLDINGS, MAX_WEIGHT, COST_PER_SIDE,
    VOL_WINDOW, COV_WINDOW, MARKET_SHARPE, TAU, VIEW_SCALE,
)


def month_starts(dates, first):
    return [i for i in range(1, len(dates))
            if dates[i].month != dates[i - 1].month and dates[i] >= pd.Timestamp(first)]


class Market:
    def __init__(self, prices, returns, caps, members, groups, copper):
        self.prices = prices
        self.returns = returns
        self.caps = caps
        self.members = members
        self.groups = groups
        self.copper = copper
        self.tickers = list(prices.columns)

    def eligible(self, i):
        window = self.prices.iloc[i - 1 - LOOKBACK:i]
        ok = self.members.iloc[i - 1] & window.notna().all() & self.caps.iloc[i - 1].notna()
        return [t for t in self.tickers if ok[t]]

    def scores(self, i, names):
        p = self.prices[names]
        return (p.iloc[i - 1 - SKIP] / p.iloc[i - 1 - LOOKBACK] - 1).sort_values()

    def winners_losers(self, i):
        s = self.scores(i, self.eligible(i))
        k = max(MIN_HOLDINGS, int(round(len(s) * TOP_FRACTION)))
        return s, list(s.index[-k:]), list(s.index[:k])

    def cap_weights(self, i, names):
        c = self.caps.iloc[i - 1][names]
        return c / c.sum()

    def copper_trend(self, i):
        c = self.copper
        return c.iloc[i - 1 - SKIP] / c.iloc[i - 1 - LOOKBACK] - 1


def to_vector(weights, columns):
    return pd.Series(weights, dtype=float).reindex(columns).fillna(0.0).values


def momentum_target(mkt, columns):
    def target(i):
        _, winners, _ = mkt.winners_losers(i)
        return to_vector({t: 1 / len(winners) for t in winners}, columns)
    return target


def winners_minus_losers(mkt, rebalances):
    R = mkt.returns
    out = pd.Series(np.nan, index=R.index)
    for n, i in enumerate(rebalances):
        end = rebalances[n + 1] if n + 1 < len(rebalances) else len(R)
        _, winners, losers = mkt.winners_losers(i)
        block = R.iloc[i + 1:end + 1]
        out.iloc[i + 1:end + 1] = block[winners].mean(axis=1).values - block[losers].mean(axis=1).values
    return out


def exposure_scale(wml, rebalances, dates):
    vols = {}
    for i in rebalances:
        past = wml.iloc[max(0, i - VOL_WINDOW):i].dropna()
        if len(past) >= VOL_WINDOW // 2:
            vols[dates[i]] = past.std(ddof=1) * np.sqrt(TRADING_DAYS)
    vols = pd.Series(vols)
    median = vols.expanding().median()
    return pd.DataFrame({"momentum_vol": vols, "scale": np.minimum(1.0, median / vols)})


def scaled_momentum_target(mkt, columns, scale):
    base = momentum_target(mkt, columns)

    def target(i):
        s = scale.get(mkt.prices.index[i], 1.0)
        w = base(i) * s
        w[columns.index("CASH")] = 1.0 - w.sum()
        return w
    return target


def covariance(mkt, i, names):
    window = mkt.returns[names].iloc[i - COV_WINDOW:i].values
    return ledoit_wolf(window)[0] * TRADING_DAYS


def black_litterman(sigma, w_mkt, P, Q):
    gamma = MARKET_SHARPE / np.sqrt(w_mkt @ sigma @ w_mkt)
    pi = gamma * sigma @ w_mkt
    ts = TAU * sigma
    omega = np.diag(np.diag(P @ ts @ P.T))
    mu = pi + ts @ P.T @ np.linalg.solve(P @ ts @ P.T + omega, Q - P @ pi)
    return mu, gamma


def optimize(mu, sigma, gamma, start):
    n = len(mu)
    objective = lambda w: -(w @ mu - 0.5 * gamma * w @ sigma @ w)
    gradient = lambda w: -(mu - gamma * sigma @ w)
    budget = [{"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(n)}]
    for x0 in [np.clip(start, 0, MAX_WEIGHT), np.full(n, 1 / n)]:
        res = minimize(objective, x0 / x0.sum(), jac=gradient, bounds=[(0, MAX_WEIGHT)] * n,
                       constraints=budget, method="SLSQP", options={"ftol": 1e-12, "maxiter": 1000})
        w = res.x
        if res.success and abs(w.sum() - 1) < 1e-6 and w.min() > -1e-6 and w.max() < MAX_WEIGHT + 1e-6:
            w = np.clip(w, 0, MAX_WEIGHT)
            return w / w.sum(), True
    return None, False


def views(mkt, i, names):
    scores, winners, losers = mkt.winners_losers(i)
    rows, q = [], []
    p = pd.Series(0.0, index=names)
    p[winners], p[losers] = 1 / len(winners), -1 / len(losers)
    rows.append(p.values)
    q.append(VIEW_SCALE * (scores[winners].mean() - scores[losers].mean()))
    trend = mkt.copper_trend(i)
    metals = [t for t in names if mkt.groups[t] == "Metals"]
    others = [t for t in names if mkt.groups[t] != "Metals"]
    if trend > 0 and metals:
        p = pd.Series(0.0, index=names)
        p[metals], p[others] = 1 / len(metals), -1 / len(others)
        rows.append(p.values)
        q.append(VIEW_SCALE * trend)
    return np.array(rows), np.array(q), trend


def black_litterman_target(mkt, columns, log):
    fallback = momentum_target(mkt, columns)

    def target(i):
        names = mkt.eligible(i)
        sigma = covariance(mkt, i, names)
        w_mkt = mkt.cap_weights(i, names).values
        P, Q, trend = views(mkt, i, names)
        mu, gamma = black_litterman(sigma, w_mkt, P, Q)
        w, ok = optimize(mu, sigma, gamma, w_mkt)
        log.append({"date": mkt.prices.index[i], "copper_trend": trend,
                    "copper_view_on": len(Q) > 1, "optimizer_ok": ok})
        return to_vector(dict(zip(names, w)), columns) if ok else fallback(i)
    return target


def benchmark_target(mkt, columns):
    def target(i):
        return to_vector(mkt.cap_weights(i, mkt.eligible(i)).to_dict(), columns)
    return target


def backtest(returns, target_fn, rebalances, cost=COST_PER_SIDE):
    columns = list(returns.columns)
    stock = np.array([c != "CASH" for c in columns])
    R = np.nan_to_num(returns.values)
    n = len(returns)
    net = np.full(n, np.nan)
    rebal = set(rebalances)
    w = None
    turnover, held = {}, {}
    for d in range(rebalances[0], n):
        g = 0.0
        if w is not None:
            g = w @ R[d]
            w = w * (1 + R[d]) / (1 + g)
        paid = 0.0
        if d in rebal:
            new = target_fn(d)
            if w is not None:
                turnover[returns.index[d]] = 0.5 * np.abs(new - w).sum()
                paid = cost * np.abs(new - w)[stock].sum()
            else:
                paid = cost * np.abs(new)[stock].sum()
            w = new
            held[returns.index[d]] = pd.Series(new, index=columns)
        net[d] = g - paid
    first = rebalances[0] + 1
    return pd.Series(net[first:], index=returns.index[first:]), pd.Series(turnover), pd.DataFrame(held).T
