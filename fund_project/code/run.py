import sys

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    TABLES, FIGURES, RISK_FREE, MARKET, COPPER, PRICE_START, FIRST_TRADE, DESIGN_END,
    HOLDOUT_START, LAST_DAY, LOOKBACK, MAX_WEIGHT, VARIANTS_TRIED,
)
from data import load, membership_matrix
from strategy import (
    Market, month_starts, momentum_target, winners_minus_losers, exposure_scale,
    scaled_momentum_target, black_litterman_target, benchmark_target, backtest,
)
from evaluate import summary, ladder, loadings

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})

NAMES = {"A": "A. Momentum", "B": "B. Crash-protected", "C": "C. Copper-confirmed BL",
         "Theme": "Theme index", "SPY": "S&P 500"}
STYLES = {"A": dict(color="black", lw=1.8), "B": dict(color="black", ls="--", lw=1.2),
          "C": dict(color="black", ls=":", lw=1.4), "Theme": dict(color="gray", lw=1.8),
          "SPY": dict(color="silver", lw=1.2)}


def build():
    total, close, shares, factors, universe = load()
    dates = total[MARKET].dropna().loc[PRICE_START:LAST_DAY].index
    tickers = universe.index.tolist()
    prices = total[tickers].reindex(dates)
    returns = prices.pct_change(fill_method=None)
    caps = close[tickers].reindex(dates) * shares[tickers].reindex(dates).ffill()
    rf = total[RISK_FREE].reindex(dates).pct_change(fill_method=None).fillna(0.0)
    spy = total[MARKET].reindex(dates).pct_change(fill_method=None)
    copper = total[COPPER].reindex(dates).ffill()
    members = membership_matrix(dates, tickers)
    mkt = Market(prices, returns, caps, members, universe["Group"], copper)

    columns = tickers + ["CASH"]
    bt_returns = returns.assign(CASH=rf)[columns]
    rebalances = month_starts(dates, FIRST_TRADE)
    history = month_starts(dates, dates[LOOKBACK + 2])

    wml = winners_minus_losers(mkt, history)
    scale = exposure_scale(wml, history, dates)
    bl_log = []

    runs = {
        "A": backtest(bt_returns, momentum_target(mkt, columns), rebalances),
        "B": backtest(bt_returns, scaled_momentum_target(mkt, columns, scale["scale"].to_dict()), rebalances),
        "C": backtest(bt_returns, black_litterman_target(mkt, columns, bl_log), rebalances),
        "Theme": backtest(bt_returns, benchmark_target(mkt, columns), rebalances),
    }
    start = runs["A"][0].index[0]
    series = {k: v[0] for k, v in runs.items()}
    series["SPY"] = spy.loc[start:]
    eligible = pd.DataFrame(
        {dates[i]: universe["Group"][mkt.eligible(i)].value_counts() for i in rebalances}).T.fillna(0)
    return dict(series=series, runs=runs, rf=rf, factors=factors, universe=universe,
                eligible=eligible, scale=scale, bl_log=pd.DataFrame(bl_log), mkt=mkt,
                rebalances=rebalances, dates=dates)


def segment(x, a, b):
    return x.loc[a:b]


def performance(res, a, b):
    out = {}
    for k in NAMES:
        r = segment(res["series"][k], a, b)
        to = segment(res["runs"][k][2], a, b) if k in res["runs"] else None
        bench = None if k in ("Theme", "SPY") else res["series"]["Theme"]
        out[NAMES[k]] = summary(r, res["rf"], to, bench)
    return pd.DataFrame(out)


def save(df, name, digits=4):
    df.round(digits).to_csv(TABLES / f"{name}.csv")


def figure_universe(res):
    e = res["eligible"]
    fig, ax = plt.subplots(figsize=(10, 4))
    for g, st in zip(["Power", "Grid equipment", "Metals"], ["-", "--", ":"]):
        ax.plot(e.index, e.get(g, 0), color="black", ls=st, label=g)
    ax.plot(e.index, e.sum(axis=1), color="gray", lw=2, label="Total")
    ax.set_ylabel("Eligible stocks")
    ax.set_title("Universe size at each monthly rebalance")
    ax.legend(frameon=False, ncol=4)
    fig.tight_layout()
    fig.savefig(FIGURES / "universe.png", dpi=200)
    plt.close(fig)


def figure_growth(res, end, name, title):
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for k in NAMES:
        r = res["series"][k].loc[:end]
        ax.plot((1 + r).cumprod(), label=NAMES[k], **STYLES[k])
    if pd.Timestamp(end) > pd.Timestamp(HOLDOUT_START):
        ax.axvspan(pd.Timestamp(HOLDOUT_START), pd.Timestamp(end), color="0.92", zorder=0)
        ax.text(pd.Timestamp("2023-02-01"), ax.get_ylim()[1] * 0.96, "Holdout", va="top")
    ax.set_ylabel("Growth of $1, net of costs")
    ax.set_title(title)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=200)
    plt.close(fig)


def figure_drawdown(res):
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for k in ["A", "B", "Theme"]:
        w = (1 + res["series"][k]).cumprod()
        ax.plot(w / w.cummax() - 1, label=NAMES[k], **STYLES[k])
    ax.axvspan(pd.Timestamp(HOLDOUT_START), pd.Timestamp(LAST_DAY), color="0.92", zorder=0)
    ax.set_ylabel("Drawdown")
    ax.set_title("Drawdowns, net of costs")
    ax.legend(frameon=False, ncol=3, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGURES / "drawdowns.png", dpi=200)
    plt.close(fig)


def figure_exposure(res, end):
    s = res["scale"].loc[FIRST_TRADE:end]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(s.index, s["scale"], color="black", lw=1.6, label="Share of capital invested (B)")
    ax2 = ax.twinx()
    ax2.plot(s.index, s["momentum_vol"], color="gray", lw=1.2, label="Momentum volatility (right)")
    ax2.spines["top"].set_visible(False)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Invested")
    ax2.set_ylabel("Annualized volatility")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax.set_title("Variant B: less capital in momentum when momentum is volatile")
    fig.tight_layout()
    fig.savefig(FIGURES / "exposure.png", dpi=200)
    plt.close(fig)


def diagnostics(res, end):
    held = res["runs"]["A"][3].loc[:end].drop(columns="CASH")
    counts = res["eligible"].loc[:end].sum(axis=1)
    log = res["bl_log"].set_index("date").loc[:end]
    return pd.Series({
        "Rebalances": len(held),
        "Eligible stocks (min)": counts.min(),
        "Eligible stocks (max)": counts.max(),
        "Stocks held by A (min)": (held > 0).sum(axis=1).min(),
        "Largest weight at rebalance (A)": held.max(axis=1).max(),
        "Largest weight at rebalance (C)": res["runs"]["C"][3].loc[:end].drop(columns="CASH").max(axis=1).max(),
        "Weight cap": MAX_WEIGHT,
        "Average invested share (B)": res["runs"]["B"][3].loc[:end].drop(columns="CASH").sum(axis=1).mean(),
        "Months with copper view on (C)": log["copper_view_on"].mean(),
        "Optimizer failures (C)": int((~log["optimizer_ok"]).sum()),
        "Variants tried": VARIANTS_TRIED,
    })


def main(mode):
    res = build()
    design = performance(res, FIRST_TRADE, DESIGN_END)
    save(design, "performance_design")
    save(res["eligible"].groupby(res["eligible"].index.year).first(), "universe_by_year", 0)
    save(diagnostics(res, DESIGN_END).to_frame("Design"), "diagnostics_design")
    figure_universe(res)
    figure_growth(res, DESIGN_END, "growth_design.png", "Design period 2016-2022, growth of $1")
    if mode == "design":
        print(design.round(3).to_string())
        print(diagnostics(res, DESIGN_END).to_string())
        return

    holdout = performance(res, HOLDOUT_START, LAST_DAY)
    full = performance(res, FIRST_TRADE, LAST_DAY)
    save(holdout, "performance_holdout")
    save(full, "performance_full")
    save(diagnostics(res, LAST_DAY).to_frame("Full"), "diagnostics_full")

    yearly = pd.DataFrame({NAMES[k]: (1 + r).groupby(r.index.year).prod() - 1
                           for k, r in res["series"].items()})
    save(yearly, "yearly_returns")

    ladders = {}
    for k in ["A", "B", "C", "Theme"]:
        for seg, (a, b) in {"Full": (FIRST_TRADE, LAST_DAY), "Holdout": (HOLDOUT_START, LAST_DAY)}.items():
            ladders[(NAMES[k], seg)] = ladder(segment(res["series"][k], a, b), res["factors"])
    save(pd.concat(ladders, names=["Strategy", "Sample", "Model"]), "attribution_ladder")

    coefs, tstats = {}, {}
    for k in ["A", "B", "C", "Theme"]:
        c, t, r2 = loadings(res["series"][k], res["factors"])
        c["R-squared"] = r2
        coefs[NAMES[k]], tstats[NAMES[k]] = c, t
    save(pd.DataFrame(coefs), "spanning_loadings_full")
    save(pd.DataFrame(tstats), "spanning_tstats_full")

    group = res["universe"]["Group"]
    tilt = {}
    for seg, (a, b) in {"Design": (FIRST_TRADE, DESIGN_END), "Holdout": (HOLDOUT_START, LAST_DAY)}.items():
        for k in ["A", "Theme"]:
            w = res["runs"][k][3].loc[a:b].drop(columns="CASH")
            tilt[(seg, NAMES[k])] = w.T.groupby(group).sum().T.mean()
    save(pd.DataFrame(tilt), "average_weight_by_group")
    held = res["runs"]["A"][3].loc[HOLDOUT_START:LAST_DAY].drop(columns="CASH")
    top = pd.DataFrame({"Months held": (held > 0).sum(), "Share of holdout months": (held > 0).mean(),
                        "Group": group}).sort_values("Months held", ascending=False).head(10)
    save(top, "most_held_holdout")

    figure_growth(res, LAST_DAY, "growth_full.png", "Growth of $1, 2016-2025 (holdout shaded)")
    figure_drawdown(res)
    figure_exposure(res, LAST_DAY)
    res["scale"].loc[FIRST_TRADE:].round(4).to_csv(TABLES / "variant_b_exposure.csv")
    res["bl_log"].to_csv(TABLES / "variant_c_log.csv", index=False)
    print(holdout.round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "final")
