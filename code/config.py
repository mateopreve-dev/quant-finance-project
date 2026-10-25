from pathlib import Path

TICKERS = [
    "AAPL", "MSFT", "CSCO", "ADBE",
    "JNJ", "PFE", "UNH",
    "JPM", "BAC", "V",
    "AMZN", "HD", "MCD",
    "GOOGL", "VZ", "NFLX",
    "BA", "HON", "UPS",
    "PG", "KO", "WMT",
    "XOM", "CVX",
    "NEE", "DUK",
    "SPG", "PLD",
    "APD", "FCX",
]

assert len(TICKERS) == 30, f"Expected 30 tickers, got {len(TICKERS)}"

START_DATE = "2016-01-01"
END_DATE = "2025-12-31"

BENCHMARK_TICKER = "^GSPC"
RISK_FREE_TICKER = "BIL"

FACTOR_NAMES = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FIGURES_DIR = PROJECT_ROOT / "output" / "figures"
OUTPUT_TABLES_DIR = PROJECT_ROOT / "output" / "tables"

for _d in [DATA_RAW_DIR, DATA_PROCESSED_DIR, OUTPUT_FIGURES_DIR, OUTPUT_TABLES_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

PRICES_CACHE = DATA_RAW_DIR / "sp500_daily_2016_2025.csv"
SHARES_CACHE = DATA_RAW_DIR / "shares_outstanding.csv"
FACTORS_CACHE = DATA_RAW_DIR / "ff_factors_daily_2016_2025.csv"
