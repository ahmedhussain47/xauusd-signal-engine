"""signal_logger.py — Enhanced signal logging with multi-level SL/TP and pips tracking."""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timezone

LOG_PATH = "results/signals_log.csv"

COLUMNS = [
    'timestamp', 'asset', 'timeframe', 'signal',
    'entry', 'sl1', 'sl2',
    'tp1', 'tp2',
    'pips_sl1', 'pips_sl2',
    'pips_tp1', 'pips_tp2',
    'rr_ratio', 'pred', 'adx', 'confidence',
    'tp_prob', 'outcome', 'exit_price',
    'pips_result', 'notes'
]


class SignalLogger:

    def __init__(self, log_path=LOG_PATH):
        self.log_path = log_path
        os.makedirs(
            os.path.dirname(log_path), exist_ok=True
        )

    def reset(self):
        """Delete all entries — fresh start"""
        pd.DataFrame(columns=COLUMNS).to_csv(
            self.log_path, index=False
        )
        print(f"Log reset: {self.log_path}")

    def _load(self):
        if not os.path.exists(self.log_path):
            pd.DataFrame(columns=COLUMNS).to_csv(
                self.log_path, index=False
            )
        return pd.read_csv(self.log_path)

    def _save(self, df):
        df.to_csv(self.log_path, index=False)

    def log_signal(self, asset, timeframe, signal,
                   entry, stop_loss, take_profit,
                   pred=0, adx=0, confidence=0,
                   tp_prob=0, timestamp=None,
                   ladder=None):

        if ladder:
            sl1 = ladder['sl1']
            sl2 = ladder['sl2']
            tp1 = ladder['tp1']
            tp2 = ladder['tp2']
            ps1 = ladder['pips_sl1']
            ps2 = ladder['pips_sl2']
            pt1 = ladder['pips_tp1']
            pt2 = ladder['pips_tp2']
        else:
            sl1 = sl2 = stop_loss
            tp1 = tp2 = take_profit
            ps1 = ps2 = pt1 = pt2 = 0

        rr = (
            abs(tp2 - entry) / abs(sl2 - entry)
            if abs(sl2 - entry) > 0 else 0
        )

        row = {
            'timestamp':   timestamp or datetime.now(
                timezone.utc
            ).strftime('%Y-%m-%d %H:%M:%S UTC'),
            'asset':       asset,
            'timeframe':   timeframe,
            'signal':      signal,
            'entry':       round(entry, 3),
            'sl1':         round(sl1, 3),
            'sl2':         round(sl2, 3),
            'tp1':         round(tp1, 3),
            'tp2':         round(tp2, 3),
            'pips_sl1':    ps1,
            'pips_sl2':    ps2,
            'pips_tp1':    pt1,
            'pips_tp2':    pt2,
            'rr_ratio':    round(rr, 2),
            'pred':        round(pred, 6),
            'adx':         round(adx, 1),
            'confidence':  confidence,
            'tp_prob':     tp_prob,
            'outcome':     '',
            'exit_price':  '',
            'pips_result': '',
            'notes':       ''
        }

        df = self._load()
        df = pd.concat(
            [df, pd.DataFrame([row])],
            ignore_index=True
        )
        self._save(df)
        print(f"Logged: {signal} {asset} @ {entry}")

    def record_outcome(self, timestamp, outcome,
                       exit_price, notes=''):
        df   = self._load()
        mask = df['timestamp'].str.contains(
            str(timestamp)[:16], na=False
        )

        if mask.sum() == 0:
            print(f"Not found: {timestamp}")
            return

        idx    = df[mask].index[-1]
        entry  = float(df.loc[idx, 'entry'])
        signal = df.loc[idx, 'signal']

        pips = (
            (exit_price - entry) * 10
            if signal == 'BUY'
            else (entry - exit_price) * 10
        )

        df.loc[idx, 'outcome']     = outcome
        df.loc[idx, 'exit_price']  = exit_price
        df.loc[idx, 'pips_result'] = round(pips, 1)
        df.loc[idx, 'notes']       = notes

        self._save(df)
        print(f"Updated: {outcome} | {pips:.0f} pips")

    def add_historical(self, trades_list):
        """Bulk add historical trades"""
        for t in trades_list:
            self.log_signal(
                asset=t.get('asset', 'XAUUSD'),
                timeframe=t.get('timeframe', '15min'),
                signal=t['signal'],
                entry=t['entry'],
                stop_loss=t.get('stop_loss', 0),
                take_profit=t.get('take_profit', 0),
                pred=t.get('pred', 0),
                adx=t.get('adx', 0),
                confidence=t.get('confidence', 0),
                tp_prob=t.get('tp_prob', 0),
                timestamp=t.get('timestamp', '')
            )
            if t.get('outcome'):
                self.record_outcome(
                    timestamp=t['timestamp'],
                    outcome=t['outcome'],
                    exit_price=t.get('exit_price', 0),
                    notes=t.get('notes', '')
                )

    def summary(self):
        df     = self._load()
        closed = df[df['outcome'] != '']

        print(f"\n{'='*45}")
        print(f"SIGNAL PERFORMANCE SUMMARY")
        print(f"{'='*45}")
        print(f"Total signals:  {len(df)}")
        print(f"Closed trades:  {len(closed)}")

        if len(closed) > 0:
            tp  = (closed['outcome'] == 'TP').sum()
            sl  = (closed['outcome'] == 'SL').sum()
            man = (closed['outcome'] == 'Manual').sum()
            wr  = tp / len(closed) * 100

            pips     = pd.to_numeric(
                closed['pips_result'], errors='coerce'
            )
            tot_pips = pips.sum()
            avg_win  = pips[pips > 0].mean()
            avg_loss = pips[pips < 0].mean()

            print(f"TP hits:        {tp}")
            print(f"SL hits:        {sl}")
            print(f"Manual:         {man}")
            print(f"Win rate:       {wr:.1f}%")
            print(f"Total pips:     {tot_pips:.0f}")
            print(f"Avg win:        {avg_win:.0f} pips")
            print(f"Avg loss:       {avg_loss:.0f} pips")

            w_sum = pips[pips > 0].sum()
            l_sum = abs(pips[pips < 0].sum())
            if l_sum > 0:
                print(f"Profit factor:  {w_sum/l_sum:.2f}")

        print(f"{'='*45}\n")
        return df
