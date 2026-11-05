# quantfinance project

## Structure

- `code/` — shared modules (`config.py`, `data_utils.py`) and one script per assignment part
- `data/raw/` — cached price data, downloaded once and reused
- `data/processed/` — derived files (returns, weights, computed series)
- `output/figures/` — saved plots
- `output/tables/` — saved result tables
- `slides/` — the .pptx deck submitted for each project

## Setup

```powershell
& "C:\pyenvs\quant-finance-project\venv\Scripts\Activate.ps1"
pip install -r requirements.txt
```

## Running

```powershell
cd code
python part1a_portfolios.py        # Project 1, Part A: VW vs EW portfolios
python part1b_diversification.py   # Project 1, Part B: diversification experiment
python part2_frontier.py           # Project 2: tangency, MVP, and the frontier
python part3_factor_covariance.py  # Project 3: sample vs. factor-based covariance, full-sample & OOS
python part4_momentum_bl.py        # Project 4: momentum views in a Black-Litterman portfolio
```

`part1a_portfolios.py` downloads and caches the 30-stock + benchmark + risk-free
price panel on first run. `part3_factor_covariance.py` additionally downloads and
caches daily Fama/French 5 factors + momentum from the Ken French Data Library
(via `pandas-datareader`) on first run.
