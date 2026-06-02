from src.signal_logger import SignalLogger
logger = SignalLogger()

# Log + Record in one go
logger.log_signal(
    asset='XAUUSD', timeframe='15min', signal='BUY',
    entry=4511.270, stop_loss=4499.529, take_profit=4534.753,
    pred=0.0001, adx=30.4, confidence=76,
    timestamp='2026-05-29 07:26'
)

logger.record_outcome(
    timestamp='2026-05-29 07:26',
    outcome='TP', exit_price=4518.270,
    notes='TP1 hit +70 pips'
)

print("\n")
logger.summary()
