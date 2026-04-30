from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "new_models"
PLOTS_DIR = RESULTS_DIR / "plots"

for path in [DATA_RAW_DIR, DATA_PROCESSED_DIR, RESULTS_DIR, PLOTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "XOM", "LLY",
    "UNH", "JNJ", "V", "PG", "MA", "HD", "COST", "ABBV", "BAC", "KO",
    "PEP", "AVGO", "WMT", "MRK", "ADBE", "CRM", "NFLX", "AMD", "INTC", "CSCO",
    "TMO", "MCD", "ABT", "DIS", "WFC", "PFE", "CVX", "ORCL", "ACN", "TXN",
    "QCOM", "LIN", "NKE", "PM", "IBM", "GE", "CAT", "GS", "MS", "BLK",
]

START_DATE = "2019-01-01"
END_DATE = None  # None means today in yfinance

# Low-compute settings. Increase later only after the pipeline works.
HORIZONS = [1, 5]
INPUT_SIZE = 128
TEST_STEP = 21
MAX_TEST_DATES = 18
MAX_STEPS = 40
FREQ = "B"
MIN_ASSETS_FOR_RANKIC = 20
RANDOM_SEED = 42

# Foundation model settings.
CHRONOS_MODEL_ID = "amazon/chronos-bolt-tiny"
KRONOS_MODEL_ID = "NeoQuasar/Kronos-small"
KRONOS_TOKENIZER_ID = "NeoQuasar/Kronos-Tokenizer-base"
KRONOS_LOCAL_REPO = PROJECT_ROOT / "models" / "Kronos"
