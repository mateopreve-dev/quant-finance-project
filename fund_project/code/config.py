from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
TABLES = ROOT / "output" / "tables"
FIGURES = ROOT / "output" / "figures"
for _d in [RAW, TABLES, FIGURES]:
    _d.mkdir(parents=True, exist_ok=True)

MEMBERSHIP_FILE = RAW / "sp500_membership.csv"
GICS_CURRENT_FILE = RAW / "sp500_gics_current.csv"
GICS_FORMER_FILE = RAW / "sp500_gics_former_members.csv"
PRICES_FILE = RAW / "prices_total_return.csv"
CLOSE_FILE = RAW / "prices_close.csv"
SHARES_FILE = RAW / "shares_outstanding_history.csv"
FACTORS_FILE = RAW / "ff5_momentum_daily.csv"
MISSING_FILE = RAW / "tickers_without_data.csv"

GROUPS = {
    "Power": ["Electric Utilities", "Multi-Utilities",
              "Independent Power Producers & Energy Traders", "Renewable Electricity"],
    "Grid equipment": ["Electrical Components & Equipment", "Heavy Electrical Equipment"],
    "Metals": ["Copper", "Steel"],
}

RISK_FREE = "BIL"
MARKET = "SPY"
COPPER = "HG=F"

PRICE_START = "2013-01-01"
PRICE_END = "2026-01-01"
FIRST_TRADE = "2016-01-01"
DESIGN_END = "2022-12-31"
HOLDOUT_START = "2023-01-01"
LAST_DAY = "2025-12-31"

TRADING_DAYS = 252
LOOKBACK = 252
SKIP = 21
TOP_FRACTION = 1 / 3
MIN_HOLDINGS = 10
MAX_WEIGHT = 0.10
COST_PER_SIDE = 0.0010
SHARES_LAG_DAYS = 63

VOL_WINDOW = 126

COV_WINDOW = 252
MARKET_SHARPE = 0.40
TAU = 1 / 60
VIEW_SCALE = 0.10

TAIL_LEVEL = 0.05
NEWEY_WEST_LAGS = 5
VARIANTS_TRIED = 3
