import pandas as pd

log_path = "results/signals_log.csv"

# Load CSV
df = pd.read_csv(log_path)

print(f"Before: {len(df)} rows")

# Delete row with timestamp matching 2026-05-29 07:26
df = df[~df['timestamp'].str.contains('2026-05-29 07:26', na=False)]

# Save
df.to_csv(log_path, index=False)

print(f"After: {len(df)} rows")
print("\n[OK] Row deleted!")

# Show summary
from src.signal_logger import SignalLogger
logger = SignalLogger()
logger.summary()
