import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import TICKERS, FACTOR_NAMES, OUTPUT_FIGURES_DIR, OUTPUT_TABLES_DIR
from data_utils import download_prices, download_factors, compute_returns

TRADING_DAYS = 252
BURN_IN_DAYS = 2 * TRADING_DAYS
ROLLING_WINDOW = TRADING_DAYS

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def sample_covariance(stock_returns, factors=None):
    return stock_returns.cov().values * TRADING_DAYS


def factor_covariance(stock_returns, factors):
    common = stock_returns.index.intersection(factors.index)
    Y = stock_returns.loc[common].values - factors.loc[common, ["RF"]].values
    F = factors.loc[common, FACTOR_NAMES].values
    X = np.column_stack([np.ones(len(common)), F])

    coeffs, *_ = np.linalg.lstsq(X, Y, rcond=None)
    B = coeffs[1:].T
    resid = Y - X @ coeffs
    resid_var = (resid ** 2).sum(axis=0) / (len(common) - X.shape[1])

    factor_cov = np.cov(F, rowvar=False)
    return (B @ factor_cov @ B.T + np.diag(resid_var)) * TRADING_DAYS


def mvp_weights(cov):
    raw = np.linalg.solve(cov, np.ones(cov.shape[0]))
    return raw / raw.sum()


def effective_n(weights):
    return 1.0 / np.sum(np.asarray(weights) ** 2)


def annualize_return(daily_returns):
    return (1 + daily_returns.mean()) ** TRADING_DAYS - 1


def annualize_vol(daily_returns):
    return daily_returns.std(ddof=1) * np.sqrt(TRADING_DAYS)


def sharpe_ratio(daily_returns, daily_rf):
    excess = daily_returns - daily_rf.reindex(daily_returns.index)
    return excess.mean() / daily_returns.std(ddof=1) * np.sqrt(TRADING_DAYS)


def first_month_start(dates, min_index):
    for i in range(max(min_index, 1), len(dates)):
        if dates[i].month != dates[i - 1].month:
            return i
    raise ValueError("Not enough data after burn-in")


def backtest(stock_returns, target_fn, start):
    dates = stock_returns.index
    R = stock_returns.values
    port = np.full(len(dates), np.nan)
    w, month, rows = None, None, []

    for i in range(start, len(dates)):
        t = dates[i]
        if (t.year, t.month) != month:
            target = target_fn(i)
            rows.append({
                "date": t,
                "max_weight": target.max(),
                "max_abs_weight": np.abs(target).max(),
                "effective_n": effective_n(target),
                "turnover": np.nan if w is None else 0.5 * np.abs(target - w).sum(),
            })
            w, month = target, (t.year, t.month)

        port[i] = w @ R[i]
        grown = w * (1 + R[i])
        w = grown / grown.sum()

    returns = pd.Series(port, index=dates).dropna()
    return returns, pd.DataFrame(rows).set_index("date")


def estimated_mvp(stock_returns, factors, estimator, window=None):
    def target(i):
        lo = 0 if window is None else max(0, i - window)
        return mvp_weights(estimator(stock_returns.iloc[lo:i], factors))
    return target


def oos_row(returns, history, rf):
    return {
        "Annualized Return": annualize_return(returns),
        "Annualized Volatility": annualize_vol(returns),
        "Sharpe Ratio": sharpe_ratio(returns, rf),
        "Avg. Max |Weight|": history["max_abs_weight"].mean(),
        "Avg. Effective N": history["effective_n"].mean(),
        "Avg. Monthly Turnover": history["turnover"].mean(),
    }


def main():
    prices = download_prices()
    stock_returns = compute_returns(prices)[TICKERS]
    factors = download_factors()

    common = stock_returns.index.intersection(factors.index)
    stock_returns = stock_returns.loc[common]
    factors = factors.loc[common]
    rf = factors["RF"]
    n = len(TICKERS)
    k = len(FACTOR_NAMES)

    print(f"Sample: {len(common)} days, {common.min().date()} to {common.max().date()}")
    print(f"Parameters: sample = {n * (n + 1) // 2}, "
          f"factor = {n * k + k * (k + 1) // 2 + n}")

    cov_sample = sample_covariance(stock_returns)
    cov_factor = factor_covariance(stock_returns, factors)
    w_sample = mvp_weights(cov_sample)
    w_factor = mvp_weights(cov_factor)

    full_rows = {}
    full_returns = {}
    for label, w in [("Sample", w_sample), ("Factor", w_factor)]:
        ret, _ = backtest(stock_returns, lambda i, w=w: w, start=0)
        full_returns[label] = ret
        full_rows[label] = {
            "Annualized Return": annualize_return(ret),
            "Annualized Volatility": annualize_vol(ret),
            "Max Weight": w.max(),
            "Min Weight": w.min(),
            "Max |Weight|": np.abs(w).max(),
            "Effective N": effective_n(w),
            "Short Positions": int((w < 0).sum()),
        }
    full_sample = pd.DataFrame(full_rows)
    print("\nFull sample\n", full_sample.round(4))
    full_sample.to_csv(OUTPUT_TABLES_DIR / "part3_full_sample_comparison.csv")

    weights = pd.DataFrame({"Sample": w_sample, "Factor": w_factor}, index=TICKERS)
    weights.to_csv(OUTPUT_TABLES_DIR / "part3_full_sample_weights.csv")

    start = first_month_start(stock_returns.index, BURN_IN_DAYS)
    specs = {
        "Sample (expanding)": (sample_covariance, None),
        "Factor (expanding)": (factor_covariance, None),
        "Sample (1-year rolling)": (sample_covariance, ROLLING_WINDOW),
        "Factor (1-year rolling)": (factor_covariance, ROLLING_WINDOW),
    }
    oos_returns, oos_history = {}, {}
    for label, (estimator, window) in specs.items():
        target = estimated_mvp(stock_returns, factors, estimator, window)
        oos_returns[label], oos_history[label] = backtest(stock_returns, target, start)

    ew = np.ones(n) / n
    oos_returns["Equal weight"], oos_history["Equal weight"] = backtest(
        stock_returns, lambda i: ew, start)

    oos = pd.DataFrame({label: oos_row(oos_returns[label], oos_history[label], rf)
                        for label in oos_returns})
    print(f"\nOut of sample, from {stock_returns.index[start].date()}\n", oos.round(4))
    oos.to_csv(OUTPUT_TABLES_DIR / "part3_oos_comparison.csv")

    oos_dates = oos_returns["Equal weight"].index
    gap = pd.DataFrame({
        label: {
            "Full-sample weights": annualize_vol(full_returns[label].loc[oos_dates]),
            "OOS expanding": annualize_vol(oos_returns[f"{label} (expanding)"]),
            "OOS 1-year rolling": annualize_vol(oos_returns[f"{label} (1-year rolling)"]),
        }
        for label in ["Sample", "Factor"]
    })
    print("\nVolatility over the OOS period\n", gap.round(4))
    gap.to_csv(OUTPUT_TABLES_DIR / "part3_insample_vs_oos.csv")

    history = pd.concat(oos_history, axis=1)
    history.to_csv(OUTPUT_TABLES_DIR / "part3_oos_concentration_history.csv")

    order = weights["Sample"].sort_values(ascending=False).index
    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - 0.2, weights.loc[order, "Sample"], 0.4, color="black", label="Sample covariance")
    ax.bar(x + 0.2, weights.loc[order, "Factor"], 0.4, color="white", edgecolor="black",
           label="Factor covariance")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=90)
    ax.set_ylabel("Weight")
    ax.set_title("Full-sample MVP weights")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_FIGURES_DIR / "part3_full_sample_weights.png", dpi=200)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for ax, kind in zip(axes, ["expanding", "1-year rolling"]):
        ax.plot(oos_history[f"Sample ({kind})"]["effective_n"], color="black",
                label="Sample covariance")
        ax.plot(oos_history[f"Factor ({kind})"]["effective_n"], color="black",
                linestyle="--", label="Factor covariance")
        ax.set_title(f"Effective number of positions, {kind} window")
        ax.set_ylabel("1 / sum(w²)")
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_FIGURES_DIR / "part3_effective_n_oos.png", dpi=200)

    fig, ax = plt.subplots(figsize=(11, 5))
    styles = {
        "Sample (expanding)": dict(color="black"),
        "Factor (expanding)": dict(color="black", linestyle="--"),
        "Equal weight": dict(color="gray"),
    }
    for label, style in styles.items():
        ax.plot((1 + oos_returns[label]).cumprod(), label=label, **style)
    ax.set_title("Out-of-sample growth of $1, monthly rebalanced")
    ax.set_ylabel("Growth of $1")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_FIGURES_DIR / "part3_oos_growth.png", dpi=200)

    plt.show()


if __name__ == "__main__":
    main()
