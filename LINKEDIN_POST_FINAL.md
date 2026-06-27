# 🎯 XAUUSD Signal Generator: 75.6% Live Accuracy vs. Academic Benchmarks

I'm proud to share a quantitative research project from **University of Koblenz** (Prof. Neuhaus) + an exceptional team.

---

## **PART 1: Research Foundation** (Collaborative)

We analyzed hundreds of financial assets (2019-2026) using statistical forecasting methods established in academia. This research laid the groundwork for understanding time-series prediction at scale.

---

## ⭐ **PART 2: Live-Trading Signal Engine** — *Solely My Work*

Following the research phase, I independently designed a **production-ready XAUUSD signal generator** to deepen my knowledge in AI-driven trading. This system now runs live with results that **significantly outperform industry benchmarks.**

### **The ML Model Testing Journey**

To achieve optimal accuracy, I tested **15 different ML model pairs** across various architectures:
- XGBoost + LightGBM combinations
- Random Forest ensembles
- LSTM + RNN neural networks
- ARIMA variations
- Other classical + deep learning approaches

**Result:** Most achieved 40-48% accuracy — *worse than random coin flips.*

This taught me a critical lesson: **Model sophistication alone doesn't drive edge.** The real alpha comes from:
✓ Intelligent signal filtering (5-tier system)
✓ Dynamic risk management (volatility scaling, ATR-based stops)
✓ Multi-timeframe bias validation
✓ Rigorous confidence scoring

---

### **Live Performance (45 Signals — Real MT5 Execution)**

✅ **75.6% win rate** vs. academic standard of 42-55%
✅ **+6,973 total pips** across all signals
✅ **+245 pips average win** | -125 pips average loss
✅ **6.08x profit factor** (wins exceed losses by 6x)
✅ **80% accuracy on high-confidence signals (≥75%)**

---

### **Architecture Overview**

📊 **Forecasting Engine:**
- 3-model weighted ensemble (AutoTheta, AutoETS, Chronos)
- Weights derived from walk-forward Sharpe ratios
- Real-time predictions 3 bars ahead (15-min timeframe)

🛡️ **Signal Filtering (5 Tiers):**
1. ADX trend strength (≥22 to avoid ranging markets)
2. RSI extremes (skip overbought/oversold)
3. 45-min cooldown between same-direction signals
4. Multi-timeframe alignment (≥2 of 4 higher TFs must agree)
5. Confidence threshold (55-95% dynamic scoring)

⚙️ **Risk Management:**
- ATR-based stop losses (1.2x multiplier)
- Volatility-scaled position sizing (75% in HIGH vol)
- Confidence-scaled position sizing (50-100%)
- Minimum R:R enforced (1.5:1)
- Multi-level TP/SL ladders

📡 **Data & Execution:**
- MT5 real-time feeds (lower latency than yfinance)
- Automated signal logging & outcome tracking
- News + sentiment bias filters
- Live market execution (not backtest)

---

### **Key Insight**

Published research on statistical forecasting models typically achieves 42-55% accuracy. This system beats that benchmark by **20+ percentage points** through disciplined filtering + risk management, not model complexity.

The lesson: **In trading, edge comes from position management, not prediction perfection.**

---

**📦 Code & Demo:**
Repo link coming soon

**📚 Thanks to:**
Prof. Neuhaus and my team for the research foundation. Part 2 was my independent work to transform academic insights into live trading alpha.

---

**#QuantitativeFinance #MachineLearning #XAUUSD #TradingSignals #LiveTrading #Ensemble #RiskManagement #FinTech**
