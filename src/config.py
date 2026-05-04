from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "new_models"
PLOTS_DIR = RESULTS_DIR / "plots"

for path in [DATA_RAW_DIR, DATA_PROCESSED_DIR, RESULTS_DIR, PLOTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

TICKERS = [
    # US Technology
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD",
    "INTC", "QCOM", "AVGO", "ADBE", "CRM", "ORCL", "CSCO", "IBM",
    "TXN", "MU", "KLAC", "LRCX", "PANW", "CRWD", "NET", "PLTR",
    "SNOW", "INTU", "NOW", "DELL", "HPQ",
    # US Financials
    "JPM", "BAC", "GS", "MS", "V", "MA", "AXP",
    "WFC", "C", "BLK", "SPGI", "MCO", "PYPL", "USB", "PNC",
    "TFC", "ICE", "CME", "BX", "KKR",
    # US Healthcare
    "JNJ", "UNH", "LLY", "PFE", "ABBV", "AMGN", "GILD", "MRK", "BMY",
    "TMO", "ABT", "MDT", "BSX", "SYK", "ISRG",
    # US Consumer
    "WMT", "PG", "KO", "PEP", "COST", "MDLZ", "MO", "PM", "EL",
    "MCD", "SBUX", "NKE", "HD", "LOW", "TGT", "BKNG", "CMG", "ORLY", "TJX",
    # US Energy & Industrials
    "XOM", "CVX", "COP", "EOG", "SLB", "PSX", "VLO", "MPC", "OXY", "HAL",
    "HON", "CAT", "DE", "BA", "GE", "MMM", "UPS", "FDX", "LMT", "RTX",
    # US Media & Telecom
    "DIS", "NFLX", "CMCSA", "T", "VZ", "SPOT", "SNAP", "ZM",
    # US Materials & Real Estate
    "LIN", "APD", "SHW", "FCX", "NEM", "ECL", "AMT", "PLD", "EQIX", "PSA", "WELL", "O",
    # US Other
    "UBER", "ABNB", "ETSY", "EBAY", "WM", "RSG",
    # Europe ADRs (USD-listed on NYSE/NASDAQ)
    "ASML", "SAP", "NVO", "AZN", "SHEL", "BP", "RIO", "BHP", "GSK", "UL", "DEO", "BTI",
    # Asia ADRs
    "TSM", "BABA", "JD", "BIDU", "NTES", "TM", "HMC", "SONY", "HSBC", "SE",
    # Canada ADRs
    "SHOP", "TD", "RY", "CNI", "CP", "ENB", "SU", "MFC",
    # Emerging Markets ADRs
    "VALE", "ITUB", "INFY", "HDB", "MELI", "PBR", "WIT", "GOLD",
    # Commodities futures
    "GC=F", "SI=F", "CL=F", "BZ=F", "NG=F", "HG=F", "PL=F", "ZC=F", "ZW=F", "ZS=F",
    # Major Forex pairs
    "EURUSD=X", "USDJPY=X", "GBPUSD=X", "AUDUSD=X", "USDCHF=X", "USDCAD=X",
    "NZDUSD=X", "EURGBP=X", "EURJPY=X", "GBPJPY=X", "AUDJPY=X", "CADJPY=X",
    "EURCHF=X", "EURAUD=X", "EURCAD=X", "GBPCHF=X", "GBPAUD=X", "GBPCAD=X",
    "AUDCAD=X", "CHFJPY=X",
]

START_DATE = "2015-01-01"
END_DATE = None  # None means today in yfinance

# Low-compute settings. Increase later only after the pipeline works.
HORIZONS = [1, 5]
INPUT_SIZE = 126
TEST_STEP = 21
MAX_TEST_DATES = 36
MAX_STEPS = 100
FREQ = "B"
MIN_ASSETS_FOR_RANKIC = 100
RANDOM_SEED = 42

# Foundation model settings.
CHRONOS_MODEL_ID = "amazon/chronos-bolt-tiny"
KRONOS_MODEL_ID = "NeoQuasar/Kronos-small"
KRONOS_TOKENIZER_ID = "NeoQuasar/Kronos-Tokenizer-base"
KRONOS_LOCAL_REPO = PROJECT_ROOT / "models" / "Kronos"
