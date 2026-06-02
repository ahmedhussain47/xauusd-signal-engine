from src.signal_logger import SignalLogger

logger = SignalLogger()

# Aaj ke confirmed trades
logger.add_historical([
    {
        'timestamp':  '2026-05-27 02:31:00 UTC',
        'asset':      'XAUUSD',
        'timeframe':  '15min',
        'signal':     'SELL',
        'entry':      4407.090,
        'stop_loss':  4421.844,
        'take_profit':4377.582,
        'adx':        25.2,
        'pred':       -0.0001,
        'outcome':    'TP',
        'exit_price': 4377.58,
        'notes':      'Strong SELL, clean TP hit'
    },
    {
        'timestamp':  '2026-05-27 05:18:00 UTC',
        'asset':      'XAUUSD',
        'timeframe':  '15min',
        'signal':     'BUY',
        'entry':      4374.130,
        'stop_loss':  4359.450,
        'take_profit':4403.489,
        'adx':        47.1,
        'pred':       0.0001,
        'outcome':    'TP',
        'exit_price': 4403.49,
        'notes':      'ADX 47, strong trend TP hit'
    },
    {
        'timestamp':  '2026-05-27 05:34:00 UTC',
        'asset':      'XAUUSD',
        'timeframe':  '15min',
        'signal':     'BUY',
        'entry':      4387.120,
        'stop_loss':  4371.760,
        'take_profit':4417.841,
        'adx':        45.5,
        'pred':       0.0001,
        'outcome':    '',
        'exit_price': 0,
        'notes':      'Still open'
    },
    {
        'timestamp':  '2026-05-27 05:51:00 UTC',
        'asset':      'XAUUSD',
        'timeframe':  '15min',
        'signal':     'BUY',
        'entry':      4393.950,
        'stop_loss':  4378.517,
        'take_profit':4424.816,
        'adx':        42.7,
        'pred':       0.0002,
        'outcome':    '',
        'exit_price': 0,
        'notes':      'Still open'
    },
    {
        'timestamp':  '2026-05-27 06:08:00 UTC',
        'asset':      'XAUUSD',
        'timeframe':  '15min',
        'signal':     'BUY',
        'entry':      4381.750,
        'stop_loss':  4366.197,
        'take_profit':4412.855,
        'adx':        41.3,
        'pred':       0.0000,
        'outcome':    '',
        'exit_price': 0,
        'notes':      'Still open'
    },
    {
        'timestamp':  '2026-05-27 06:16:00 UTC',
        'asset':      'XAUUSD',
        'timeframe':  '15min',
        'signal':     'SELL',
        'entry':      4374.670,
        'stop_loss':  4389.982,
        'take_profit':4344.047,
        'adx':        41.1,
        'pred':       -0.0003,
        'outcome':    'SL',
        'exit_price': 4389.98,
        'notes':      'Wrong direction, price recovered'
    },
    {
        'timestamp':  '2026-05-27 06:33:00 UTC',
        'asset':      'XAUUSD',
        'timeframe':  '15min',
        'signal':     'SELL',
        'entry':      4383.800,
        'stop_loss':  4399.507,
        'take_profit':4352.386,
        'adx':        40.4,
        'pred':       -0.0002,
        'outcome':    'SL',
        'exit_price': 4399.51,
        'notes':      'Wrong direction, SL hit'
    },
    {
        'timestamp':  '2026-05-27 06:49:00 UTC',
        'asset':      'XAUUSD',
        'timeframe':  '15min',
        'signal':     'SELL',
        'entry':      4383.350,
        'stop_loss':  4398.770,
        'take_profit':4352.509,
        'adx':        39.7,
        'pred':       -0.0001,
        'outcome':    'SL',
        'exit_price': 4398.77,
        'notes':      'Wrong direction, SL hit'
    },
])

print("\n[OK] Trades added!\n")
logger.summary()
