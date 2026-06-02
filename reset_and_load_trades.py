from src.signal_logger import SignalLogger

logger = SignalLogger()

# Delete all old entries
logger.reset()

# Add today's 23 trades
logger.add_historical([
    # WINS (17 TP)
    {'timestamp': '2026-05-27 02:31', 'signal': 'SELL',
     'entry': 4407.09, 'stop_loss': 4421.844,
     'take_profit': 4377.582, 'adx': 25.2,
     'outcome': 'TP', 'exit_price': 4377.58},

    {'timestamp': '2026-05-27 05:18', 'signal': 'BUY',
     'entry': 4374.13, 'stop_loss': 4359.45,
     'take_profit': 4403.489, 'adx': 47.1,
     'outcome': 'TP', 'exit_price': 4403.49},

    {'timestamp': '2026-05-27 05:34', 'signal': 'BUY',
     'entry': 4387.12, 'stop_loss': 4371.76,
     'take_profit': 4417.841, 'adx': 45.5,
     'outcome': 'TP', 'exit_price': 4417.84},

    {'timestamp': '2026-05-27 05:51', 'signal': 'BUY',
     'entry': 4393.95, 'stop_loss': 4378.517,
     'take_profit': 4424.816, 'adx': 42.7,
     'outcome': 'TP', 'exit_price': 4424.82},

    {'timestamp': '2026-05-27 06:08', 'signal': 'BUY',
     'entry': 4381.75, 'stop_loss': 4366.197,
     'take_profit': 4412.855, 'adx': 41.3,
     'outcome': 'TP', 'exit_price': 4412.86},

    {'timestamp': '2026-05-27 07:06', 'signal': 'BUY',
     'entry': 4388.88, 'stop_loss': 4373.883,
     'take_profit': 4418.874, 'adx': 38.3,
     'outcome': 'TP', 'exit_price': 4418.87},

    {'timestamp': '2026-05-27 07:15', 'signal': 'BUY',
     'entry': 4388.98, 'stop_loss': 4374.737,
     'take_profit': 4417.467, 'adx': 36.2,
     'outcome': 'TP', 'exit_price': 4417.47},

    {'timestamp': '2026-05-27 07:32', 'signal': 'BUY',
     'entry': 4390.94, 'stop_loss': 4376.454,
     'take_profit': 4419.912, 'adx': 33.6,
     'outcome': 'TP', 'exit_price': 4419.91},

    {'timestamp': '2026-05-27 07:48', 'signal': 'BUY',
     'entry': 4387.28, 'stop_loss': 4372.784,
     'take_profit': 4416.272, 'adx': 31.9,
     'outcome': 'TP', 'exit_price': 4416.27},

    {'timestamp': '2026-05-27 08:05', 'signal': 'BUY',
     'entry': 4386.84, 'stop_loss': 4372.341,
     'take_profit': 4415.839, 'adx': 30.6,
     'outcome': 'TP', 'exit_price': 4415.84},

    {'timestamp': '2026-05-27 08:22', 'signal': 'BUY',
     'entry': 4393.34, 'stop_loss': 4379.292,
     'take_profit': 4421.435, 'adx': 28.5,
     'outcome': 'TP', 'exit_price': 4421.44},

    {'timestamp': '2026-05-27 09:20', 'signal': 'BUY',
     'entry': 4391.70, 'stop_loss': 4378.853,
     'take_profit': 4417.394, 'adx': 23.0,
     'outcome': 'TP', 'exit_price': 4417.39},

    {'timestamp': '2026-05-27 09:37', 'signal': 'BUY',
     'entry': 4394.81, 'stop_loss': 4381.630,
     'take_profit': 4421.170, 'adx': 22.3,
     'outcome': 'TP', 'exit_price': 4421.17},

    {'timestamp': '2026-05-27 09:45', 'signal': 'BUY',
     'entry': 4390.33, 'stop_loss': 4377.768,
     'take_profit': 4415.454, 'adx': 21.3,
     'outcome': 'TP', 'exit_price': 4415.45},

    {'timestamp': '2026-05-27 10:02', 'signal': 'BUY',
     'entry': 4392.29, 'stop_loss': 4379.901,
     'take_profit': 4417.068, 'adx': 20.4,
     'outcome': 'TP', 'exit_price': 4417.07},

    {'timestamp': '2026-05-27 10:19', 'signal': 'BUY',
     'entry': 4398.66, 'stop_loss': 4385.882,
     'take_profit': 4424.216, 'adx': 21.6,
     'outcome': 'TP', 'exit_price': 4424.22},

    {'timestamp': '2026-05-27 10:35', 'signal': 'BUY',
     'entry': 4398.77, 'stop_loss': 4385.812,
     'take_profit': 4424.687, 'adx': 21.6,
     'outcome': 'TP', 'exit_price': 4424.69},

    # LOSSES (6 SL)
    {'timestamp': '2026-05-27 06:16', 'signal': 'SELL',
     'entry': 4374.67, 'stop_loss': 4389.982,
     'take_profit': 4344.047, 'adx': 41.1,
     'outcome': 'SL', 'exit_price': 4389.98},

    {'timestamp': '2026-05-27 06:33', 'signal': 'SELL',
     'entry': 4383.80, 'stop_loss': 4399.507,
     'take_profit': 4352.386, 'adx': 40.4,
     'outcome': 'SL', 'exit_price': 4399.51},

    {'timestamp': '2026-05-27 06:49', 'signal': 'SELL',
     'entry': 4383.35, 'stop_loss': 4398.770,
     'take_profit': 4352.509, 'adx': 39.7,
     'outcome': 'SL', 'exit_price': 4398.77},

    {'timestamp': '2026-05-27 08:30', 'signal': 'SELL',
     'entry': 4390.14, 'stop_loss': 4403.671,
     'take_profit': 4363.079, 'adx': 26.8,
     'outcome': 'SL', 'exit_price': 4403.67},

    {'timestamp': '2026-05-27 08:47', 'signal': 'SELL',
     'entry': 4388.21, 'stop_loss': 4401.815,
     'take_profit': 4361.000, 'adx': 25.6,
     'outcome': 'SL', 'exit_price': 4401.82},

    {'timestamp': '2026-05-27 09:04', 'signal': 'SELL',
     'entry': 4390.13, 'stop_loss': 4403.560,
     'take_profit': 4363.271, 'adx': 24.4,
     'outcome': 'SL', 'exit_price': 4403.56},
])

print("\n")
logger.summary()
