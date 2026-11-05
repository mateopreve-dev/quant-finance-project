import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.covariance import ledoit_wolf

from config import TICKERS, RISK_FREE_TICKER, OUTPUT_FIGURES_DIR, OUTPUT_TABLES_DIR
from data_utils import download_prices, download_shares_outstanding, compute_returns

TRADING_DAYS = 252
LOOKBACK = 252
SKIP = 21
COV_WINDOW = 252
MARKET_SHARPE = 0.40
TAU = 1 / 60
VIEW_SCALE = 0.10
CONFIDENCE = 1.0
MAX_WEIGHT = 0.25
COST_PER_SIDE = 0.0010
GROUP_SIZE = 10
DECILE_SIZE = 3

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def rebalance_dates(dates, start):
    return [i for i in range(max(start, 1), len(dates)) if dates[i].month != dates[i - 1].month]


def momentum_scores(prices, i):
    return prices.iloc[i - 1 - SKIP] / prices.iloc[i - 1 - LOOKBACK] - 1


def covariance(returns, i):
    window = returns.iloc[i - COV_WINDOW:i].values
    shrunk, _ = ledoit_wolf(window)
    return shrunk * TRADING_DAYS


def market_weights(prices, shares, i):
    caps = prices.iloc[i - 1] * shares
    return (caps / caps.sum()).values


def momentum_view(scores):
    order = scores.sort_values()
    losers, winners = order.index[:GROUP_SIZE], order.index[-GROUP_SIZE:]
    P = pd.Series(0.0, index=scores.index)
    P[winners] = 1 / GROUP_SIZE
    P[losers] = -1 / GROUP_SIZE
    spread = order[winners].mean() - order[losers].mean()
    return P.values.reshape(1, -1), np.array([VIEW_SCALE * spread])


def black_litterman(sigma, w_mkt, P, Q, confidence=CONFIDENCE):
    gamma = MARKET_SHARPE / np.sqrt(w_mkt @ sigma @ w_mkt)
    pi = gamma * sigma @ w_mkt
    omega = P @ (TAU * sigma) @ P.T / confidence
    ts = TAU * sigma
    mu = pi + ts @ P.T @ np.linalg.solve(P @ ts @ P.T + omega, Q - P @ pi)
    return mu, gamma


def optimize(mu, sigma, gamma, start):
    n = len(mu)
    result = minimize(
        lambda w: -(w @ mu - 0.5 * gamma * w @ sigma @ w),
        np.clip(start, 0, MAX_WEIGHT) / np.clip(start, 0, MAX_WEIGHT).sum(),
        jac=lambda w: -(mu - gamma * sigma @ w),
        bounds=[(0, MAX_WEIGHT)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(n)}],
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 500},
    )
    return result.x


def backtest(returns, target_fn, dates_idx, cost=COST_PER_SIDE):
    R = returns.values
    n_days = len(returns)
    gross = np.full(n_days, np.nan)
    net = np.full(n_days, np.nan)
    rebal = set(dates_idx)
    w, turnover, rows = None, [], []

    for i in range(dates_idx[0], n_days):
        trade_cost = 0.0
        if i in rebal:
            target = target_fn(i)
            traded = np.abs(target).sum() if w is None else np.abs(target - w).sum()
            turnover.append(0.5 * traded if w is not None else np.nan)
            trade_cost = cost * traded
            w = target
            rows.append(pd.Series(target, index=returns.columns, name=returns.index[i]))
        gross[i] = w @ R[i]
        net[i] = gross[i] - trade_cost
        long_value = w * (1 + R[i])
        w = long_value / (1 + gross[i])

    index = returns.index
    return (pd.Series(gross, index=index).dropna(), pd.Series(net, index=index).dropna(),
            pd.Series(turnover[1:], index=[r.name for r in rows[1:]]), pd.DataFrame(rows))


def long_short_backtest(returns, prices, dates_idx, cost=COST_PER_SIDE):
    def target(i):
        order = momentum_scores(prices, i).sort_values()
        w = pd.Series(0.0, index=returns.columns)
        w[order.index[-DECILE_SIZE:]] = 1 / DECILE_SIZE
        w[order.index[:DECILE_SIZE]] = -1 / DECILE_SIZE
        return w.values

    R = returns.values
    n_days = len(returns)
    gross = np.full(n_days, np.nan)
    net = np.full(n_days, np.nan)
    legs = np.full((n_days, 2), np.nan)
    rebal = set(dates_idx)
    long_w = short_w = None
    turnover = []

    for i in range(dates_idx[0], n_days):
        trade_cost = 0.0
        if i in rebal:
            t = target(i)
            new_long, new_short = np.clip(t, 0, None), np.clip(-t, 0, None)
            if long_w is None:
                traded = new_long.sum() + new_short.sum()
            else:
                traded = np.abs(new_long - long_w).sum() + np.abs(new_short - short_w).sum()
                turnover.append(0.5 * traded / 2)
            trade_cost = cost * traded
            long_w, short_w = new_long, new_short
        long_ret = long_w @ R[i]
        short_ret = short_w @ R[i]
        gross[i] = long_ret - short_ret
        net[i] = gross[i] - trade_cost
        legs[i] = long_ret, short_ret
        long_w = long_w * (1 + R[i]) / (1 + long_ret)
        short_w = short_w * (1 + R[i]) / (1 + short_ret)

    index = returns.index
    legs = pd.DataFrame(legs, index=index, columns=["Winners", "Losers"]).dropna()
    return (pd.Series(gross, index=index).dropna(), pd.Series(net, index=index).dropna(),
            pd.Series(turnover), legs)


def max_drawdown(daily):
    wealth = (1 + daily).cumprod()
    return (wealth / wealth.cummax() - 1).min()


def stats(daily, rf, turnover, excess=True):
    rf = rf.reindex(daily.index).fillna(0)
    ex = daily - rf if excess else daily
    return {
        "Annualized Return": (1 + daily).prod() ** (TRADING_DAYS / len(daily)) - 1,
        "Annualized Volatility": daily.std(ddof=1) * np.sqrt(TRADING_DAYS),
        "Sharpe Ratio": ex.mean() / daily.std(ddof=1) * np.sqrt(TRADING_DAYS),
        "Maximum Drawdown": max_drawdown(daily),
        "Annualized Turnover": turnover.mean() * 12 if len(turnover) else 0.0,
    }


def active_stats(daily, bench):
    active = daily - bench.reindex(daily.index)
    te = active.std(ddof=1) * np.sqrt(TRADING_DAYS)
    return te, active.mean() * TRADING_DAYS / te


def main():
    prices_all = download_prices()
    shares = download_shares_outstanding().reindex(TICKERS)
    returns_all = compute_returns(prices_all)
    prices = prices_all[TICKERS].loc[returns_all.index]
    returns = returns_all[TICKERS]
    rf = returns_all[RISK_FREE_TICKER]

    dates = rebalance_dates(returns.index, max(LOOKBACK + 1, COV_WINDOW))

    cache = {}

    def inputs(i):
        if i not in cache:
            sigma = covariance(returns, i)
            w_mkt = market_weights(prices, shares, i)
            P, Q = momentum_view(momentum_scores(prices, i))
            cache[i] = (sigma, w_mkt, P, Q)
        return cache[i]

    def bl_target(confidence=CONFIDENCE):
        def target(i):
            sigma, w_mkt, P, Q = inputs(i)
            mu, gamma = black_litterman(sigma, w_mkt, P, Q, confidence)
            return optimize(mu, sigma, gamma, w_mkt)
        return target

    def cap_target(i):
        return inputs(i)[1]

    results = {}
    bl_gross, bl_net, bl_to, bl_weights = backtest(returns, bl_target(), dates)
    results["BL momentum"] = (bl_net, bl_to)
    _, cap_net, cap_to, _ = backtest(returns, cap_target, dates)
    results["Market-cap benchmark"] = (cap_net, cap_to)
    _, ls_net, ls_to, ls_legs = long_short_backtest(returns, prices, dates)

    perf = {label: stats(ret, rf, to) for label, (ret, to) in results.items()}
    perf["Naive momentum (top-bottom decile)"] = stats(ls_net, rf, ls_to, excess=False)
    perf = pd.DataFrame(perf)
    te, ir = active_stats(bl_net, cap_net)
    perf.loc["Tracking Error vs Cap", "BL momentum"] = te
    perf.loc["Information Ratio vs Cap", "BL momentum"] = ir
    perf.loc["Gross Annualized Return", "BL momentum"] = (1 + bl_gross).prod() ** (TRADING_DAYS / len(bl_gross)) - 1
    print("\nPerformance, net of 10 bp per side\n", perf.round(4))
    perf.to_csv(OUTPUT_TABLES_DIR / "part4_performance.csv")

    yearly = pd.DataFrame({
        "BL momentum": (1 + bl_net).groupby(bl_net.index.year).prod() - 1,
        "Market-cap benchmark": (1 + cap_net).groupby(cap_net.index.year).prod() - 1,
    })
    yearly["Difference"] = yearly["BL momentum"] - yearly["Market-cap benchmark"]
    print("\nCalendar-year returns\n", yearly.round(4))
    yearly.to_csv(OUTPUT_TABLES_DIR / "part4_yearly_returns.csv")

    leg_means = ls_legs.mean() * TRADING_DAYS
    print("\nNaive momentum legs, average annual return\n", leg_means.round(4))
    leg_means.rename("Average annual return").to_csv(OUTPUT_TABLES_DIR / "part4_naive_legs.csv")
    monthly = (1 + ls_net).groupby([ls_net.index.year, ls_net.index.month]).prod() - 1
    worst = monthly.nsmallest(5).rename("Naive momentum monthly return")
    print("\nWorst months for naive momentum\n", worst.round(4))
    worst.to_csv(OUTPUT_TABLES_DIR / "part4_naive_worst_months.csv")

    sens = {}
    for c in [0.25, 1.0, 4.0]:
        _, net, to, _ = backtest(returns, bl_target(confidence=c), dates)
        te, ir = active_stats(net, cap_net)
        row = stats(net, rf, to)
        row.update({"Tracking Error vs Cap": te, "Information Ratio vs Cap": ir})
        sens[f"Confidence x{c:g}"] = row
    sens = pd.DataFrame(sens)
    print("\nSensitivity to view confidence\n", sens.round(4))
    sens.to_csv(OUTPUT_TABLES_DIR / "part4_confidence_sensitivity.csv")

    active = bl_weights.values - np.vstack([inputs(i)[1] for i in dates])
    active = pd.DataFrame(active, index=bl_weights.index, columns=TICKERS)
    tilt = []
    for k, i in enumerate(dates):
        order = momentum_scores(prices, i).sort_values().index
        tilt.append({
            "date": returns.index[i],
            "Winners": active.iloc[k][order[-GROUP_SIZE:]].sum(),
            "Losers": active.iloc[k][order[:GROUP_SIZE]].sum(),
        })
    tilt = pd.DataFrame(tilt).set_index("date")
    print("\nAverage active weight by momentum group\n", tilt.mean().round(4))
    tilt.to_csv(OUTPUT_TABLES_DIR / "part4_active_tilt.csv")

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot((1 + bl_net).cumprod(), color="black", label="BL momentum")
    ax.plot((1 + cap_net).cumprod(), color="black", linestyle="--", label="Market-cap benchmark")
    ax.plot((1 + ls_net).cumprod(), color="gray", label="Naive momentum (long-short)")
    ax.set_title("Growth of $1, net of costs")
    ax.set_ylabel("Growth of $1")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_FIGURES_DIR / "part4_growth.png", dpi=200)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(tilt["Winners"], color="black", label="Top 10 by momentum")
    ax.plot(tilt["Losers"], color="black", linestyle="--", label="Bottom 10 by momentum")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_title("Active weight versus market-cap weights")
    ax.set_ylabel("Active weight")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_FIGURES_DIR / "part4_active_tilt.png", dpi=200)


if __name__ == "__main__":
    main()
