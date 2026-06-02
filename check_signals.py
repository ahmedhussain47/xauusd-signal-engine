import pandas as pd

df = pd.read_csv('results/signals_log.csv')

print(f'Total signals: {len(df)}')
print(f'BUY: {(df["signal"] == "BUY").sum()}  |  SELL: {(df["signal"] == "SELL").sum()}')
print()
print('Last 10 signals:')
print(df[['timestamp', 'signal', 'entry', 'take_profit', 'stop_loss', 'model_pred']].tail(10).to_string())
print()
print('Resolved outcomes:')
resolved = df[df['outcome'].notna()]
print(f'  TP: {(resolved["outcome"] == "TP").sum()}')
print(f'  SL: {(resolved["outcome"] == "SL").sum()}')
print(f'  EXPIRED: {(resolved["outcome"] == "EXPIRED").sum()}')
if len(resolved) > 0:
    print(f'  Win rate: {(resolved["outcome"] == "TP").sum() / len(resolved) * 100:.1f}%')
else:
    print('  (No resolved outcomes yet)')
