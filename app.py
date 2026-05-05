"""
Gold Trade Planner — Streamlit Web App
3-model ensemble (AutoTheta + AutoETS), multi-timeframe alignment, risk management,
live 1-min gold chart with auto-refresh.
"""

import time
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsforecast import StatsForecast
from statsforecast.models import AutoTheta, AutoETS
from datetime import datetime, timezone

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Gold Trade Planner",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
TICKER     = "GC=F"
TRAIN_BARS = 200

_W  = {"AutoTheta": 0.771, "AutoETS": 0.162}
_WS = sum(_W.values())
W_THETA, W_ETS = _W["AutoTheta"] / _WS, _W["AutoETS"] / _WS

TIMEFRAMES = {
    "1 min":  {"interval": "1m",  "period": "1d",  "season": 60,  "freq": "min"},
    "5 min":  {"interval": "5m",  "period": "5d",  "season": 78,  "freq": "5min"},
    "15 min": {"interval": "15m", "period": "60d", "season": 26,  "freq": "15min"},
    "1 hour": {"interval": "60m", "period": "60d", "season": 24,  "freq": "h"},
    "4 hour": {"interval": "60m", "period": "60d", "season": 6,   "freq": "h",  "resample": "4h"},
    "Daily":  {"interval": "1d",  "period": "2y",  "season": 5,   "freq": "D"},
}

_CHART_BASE = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif"),
    xaxis_rangeslider_visible=False,
    legend=dict(orientation="h", y=1.02, bgcolor="rgba(0,0,0,0)"),
    margin=dict(l=0, r=130, t=30, b=0),
)


# ── Indicators ────────────────────────────────────────────────────────────────

def compute_atr(df, period=14):
    hi, lo, cl = df["high"], df["low"], df["close"]
    tr = pd.concat(
        [hi - lo, (hi - cl.shift()).abs(), (lo - cl.shift()).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def compute_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def compute_rsi(series, period=14):
    d    = series.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-d).clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    rs   = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def compute_adx(df, period=14):
    hi, lo, cl = df["high"], df["low"], df["close"]
    pdm = hi.diff().clip(lower=0)
    ndm = (-lo.diff()).clip(lower=0)
    pdm[pdm < ndm] = 0
    ndm[ndm < pdm] = 0
    tr    = pd.concat([hi - lo, (hi - cl.shift()).abs(), (lo - cl.shift()).abs()], axis=1).max(axis=1)
    atr14 = tr.ewm(alpha=1 / period, adjust=False).mean()
    pdi   = 100 * pdm.ewm(alpha=1 / period, adjust=False).mean() / atr14
    ndi   = 100 * ndm.ewm(alpha=1 / period, adjust=False).mean() / atr14
    dx    = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    adx   = dx.ewm(alpha=1 / period, adjust=False).mean()
    return float(adx.iloc[-1]), float(pdi.iloc[-1]), float(ndi.iloc[-1])


# ── Data & forecast ───────────────────────────────────────────────────────────

@st.cache_data(ttl=60, show_spinner=False)
def fetch_data(interval: str, period: str, resample: str | None = None) -> pd.DataFrame:
    df = yf.Ticker(TICKER).history(interval=interval, period=period)
    if df.empty:
        raise ValueError(f"yfinance returned no data (interval={interval})")
    df.columns = [c.lower() for c in df.columns]
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep].dropna()
    if not df.index.tz:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    if resample:
        df = df.resample(resample).agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna()
    if df.empty:
        raise ValueError(f"DataFrame empty after processing (interval={interval})")
    return df


def run_forecast(df: pd.DataFrame, season: int, freq: str, h: int = 4):
    log_ret = np.log(df["close"] / df["close"].shift(1)).dropna()
    series  = log_ret.iloc[-TRAIN_BARS:].values.astype(float)
    n       = len(series)
    sf_df   = pd.DataFrame({
        "unique_id": ["gold"] * n,
        "ds": pd.date_range("2000-01-01", periods=n, freq=freq),
        "y": series,
    })
    preds = StatsForecast(
        models=[AutoTheta(season_length=season), AutoETS(season_length=season)],
        freq=freq, n_jobs=1,
    ).forecast(df=sf_df, h=h)
    theta = preds["AutoTheta"].values
    ets   = preds["AutoETS"].values
    return theta, ets, W_THETA * theta + W_ETS * ets


# ── Signal helpers ────────────────────────────────────────────────────────────

def compute_confidence(direction, rsi, adx_val, pdi, ndi, ema20, cur_px, tf_dirs):
    score = 30                                   # conservative baseline (was 50)
    if adx_val > 25:   score += 10
    elif adx_val > 20: score += 5
    if direction == "BUY"  and pdi > ndi:        score += 10
    if direction == "SELL" and ndi > pdi:        score += 10
    if direction == "BUY"  and 40 < rsi < 65:   score += 8
    if direction == "SELL" and 35 < rsi < 60:   score += 8
    if direction == "SELL" and cur_px < ema20:   score += 7
    if direction == "BUY"  and cur_px > ema20:   score += 7
    aligned = sum(1 for d in tf_dirs.values() if d == direction)
    score  += int((aligned / max(len(tf_dirs), 1)) * 15)
    return min(max(score, 0), 100)


def compute_entry(direction, cur_px, ema20):
    if direction == "SELL":
        if cur_px < ema20:
            return round(ema20, 2), "LIMIT — Sell at EMA20 (resistance)"
        return round(cur_px, 2), "MARKET — Sell at current close"
    if cur_px > ema20:
        return round(ema20, 2), "LIMIT — Buy at EMA20 (support)"
    return round(cur_px, 2), "MARKET — Buy at current close"


def compute_sl_tp(df, direction, entry, atr14, swing_bars, atr_mult, rr_ratio):
    s_high = float(df["high"].tail(swing_bars).max())
    s_low  = float(df["low"].tail(swing_bars).min())
    atr_sl = atr14 * atr_mult
    if direction == "SELL":
        sl      = round(max(entry + atr_sl, s_high + atr14 * 0.3), 2)
        sl_dist = sl - entry
        tp      = round(max(entry - sl_dist * rr_ratio, s_low - atr14 * 0.2), 2)
    else:
        sl      = round(min(entry - atr_sl, s_low - atr14 * 0.3), 2)
        sl_dist = entry - sl
        tp      = round(min(entry + sl_dist * rr_ratio, s_high + atr14 * 0.2), 2)
    return sl, tp, abs(sl_dist)


# ── Chart builder ─────────────────────────────────────────────────────────────

def build_price_chart(df, ema20_s, ema50_s, rsi_s, entry=None, sl=None, tp=None,
                      title="XAUUSD", lookback=120):
    chart_df = df.tail(lookback)
    ema20_s  = ema20_s.tail(lookback)
    ema50_s  = ema50_s.tail(lookback)
    rsi_s    = rsi_s.tail(lookback)

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.60, 0.22, 0.18],
        vertical_spacing=0.03,
        subplot_titles=[title, "RSI(14)", "Volume"],
    )

    fig.add_trace(go.Candlestick(
        x=chart_df.index,
        open=chart_df["open"], high=chart_df["high"],
        low=chart_df["low"],   close=chart_df["close"],
        name="Price",
        increasing_line_color="#00C853", increasing_fillcolor="#00C853",
        decreasing_line_color="#FF1744", decreasing_fillcolor="#FF1744",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=chart_df.index, y=ema20_s, name="EMA20",
        line=dict(color="#FFA726", width=1.6)), row=1, col=1)
    fig.add_trace(go.Scatter(x=chart_df.index, y=ema50_s, name="EMA50",
        line=dict(color="#7B8CDE", width=1.4)), row=1, col=1)

    if entry is not None:
        for level, color, label in [
            (entry, "#FFA726", f"Entry  ${entry:,.2f}"),
            (sl,    "#FF1744", f"SL      ${sl:,.2f}"),
            (tp,    "#00C853", f"TP      ${tp:,.2f}"),
        ]:
            fig.add_shape(type="line",
                          x0=chart_df.index[0], x1=chart_df.index[-1],
                          y0=level, y1=level,
                          line=dict(color=color, dash="dash", width=1.5),
                          row=1, col=1)
            fig.add_annotation(x=chart_df.index[-1], y=level,
                               text=f" {label}", xanchor="left", showarrow=False,
                               font=dict(color=color, size=11), row=1, col=1)

    rsi_arr = rsi_s.values
    fig.add_trace(go.Scatter(x=chart_df.index, y=rsi_arr, name="RSI",
        line=dict(color="#CE93D8", width=1.5)), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=list(chart_df.index) + list(chart_df.index[::-1]),
        y=[70] * len(chart_df) + list(rsi_arr[::-1]),
        fill="toself", fillcolor="rgba(255,23,68,0.08)", line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=list(chart_df.index) + list(chart_df.index[::-1]),
        y=[30] * len(chart_df) + list(rsi_arr[::-1]),
        fill="toself", fillcolor="rgba(0,200,83,0.08)", line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ), row=2, col=1)
    for lvl, clr in [(70, "#FF1744"), (30, "#00C853"), (50, "#444")]:
        fig.add_hline(y=lvl, line=dict(color=clr, dash="dot", width=1), row=2, col=1)

    vol_colors = ["#00C853" if c >= o else "#FF1744"
                  for c, o in zip(chart_df["close"], chart_df["open"])]
    fig.add_trace(go.Bar(x=chart_df.index, y=chart_df["volume"],
        name="Volume", marker_color=vol_colors, opacity=0.6), row=3, col=1)

    fig.update_layout(height=720, **_CHART_BASE)
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1, gridcolor="#333")
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100], gridcolor="#333")
    fig.update_yaxes(title_text="Vol",  row=3, col=1, gridcolor="#333")
    fig.update_xaxes(gridcolor="#333", showgrid=False)
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## Settings")
    entry_tf   = st.selectbox("Entry Timeframe", list(TIMEFRAMES.keys()), index=2)
    st.divider()
    st.markdown("**Risk Management**")
    account    = st.number_input("Account Balance (USD)", 100, 1_000_000, 1_000, 100)
    risk_pct   = st.slider("Risk per Trade (%)", 0.5, 5.0, 1.0, 0.5) / 100
    rr_ratio   = st.slider("R:R Ratio", 1.0, 5.0, 2.0, 0.5)
    atr_mult   = st.slider("ATR Stop Loss Mult", 1.0, 3.0, 1.5, 0.25)
    swing_bars = st.slider("Swing Lookback (bars)", 10, 50, 20, 5)
    st.divider()
    live_refresh = st.select_slider(
        "Live Chart Refresh (s)", options=[15, 30, 60, 120, 300], value=30
    )
    st.caption("Signal data cached 60s. Live chart auto-refreshes.")


# ── Page header ───────────────────────────────────────────────────────────────

st.title("Gold Trade Planner")
st.markdown(
    "**Developed by Ahmed R. Hussain** &nbsp;|&nbsp; "
    "*Beta — For research purposes only. Not financial advice.*"
)

tab_signal, tab_live = st.tabs(["📊 Signal Generator", "📡 Live 1-Min Chart"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Signal Generator
# ═══════════════════════════════════════════════════════════════════════════════

with tab_signal:

    st.caption(
        f"AutoTheta ({W_THETA:.0%}) + AutoETS ({W_ETS:.0%}) ensemble  |  "
        f"Live via yfinance  |  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )

    generate = st.button("⚡ Generate Signal", type="primary", use_container_width=True)

    # Compute and store in session_state when button clicked
    if generate:
        with st.spinner("Fetching live Gold data and running models..."):
            cfg = TIMEFRAMES[entry_tf]
            try:
                df = fetch_data(cfg["interval"], cfg["period"], cfg.get("resample"))
            except Exception as e:
                st.error(f"Failed to fetch Gold data: {e}")
                st.stop()

            cur_px  = float(df["close"].iloc[-1])
            atr14   = float(compute_atr(df, 14).iloc[-1])
            ema20_v = float(compute_ema(df["close"], 20).iloc[-1])
            ema50_v = float(compute_ema(df["close"], 50).iloc[-1])
            rsi14   = float(compute_rsi(df["close"], 14).iloc[-1])
            adx_val, pdi, ndi = compute_adx(df, 14)

            theta, ets, ens = run_forecast(df, cfg["season"], cfg["freq"], h=4)
            pred_return = float(ens[0])

            # Gate: model must predict at least 0.02% move — below this is noise
            MIN_PRED_RETURN = 0.0002
            if abs(pred_return) < MIN_PRED_RETURN:
                st.warning(
                    f"**NO SIGNAL** — Model predicted return is too small "
                    f"({pred_return*100:+.4f}%). "
                    f"Threshold: ±{MIN_PRED_RETURN*100:.2f}%. Market likely ranging."
                )
                st.stop()

            direction = "SELL" if pred_return < 0 else "BUY"

            sweep_tfs = {k: v for k, v in TIMEFRAMES.items() if k != "1 min"}
            tf_dirs, tf_returns = {}, {}
            for tf_name, tcfg in sweep_tfs.items():
                try:
                    df_tf = fetch_data(tcfg["interval"], tcfg["period"], tcfg.get("resample"))
                    _, _, e = run_forecast(df_tf, tcfg["season"], tcfg["freq"], h=1)
                    tf_dirs[tf_name]    = "SELL" if e[0] < 0 else "BUY"
                    tf_returns[tf_name] = float(e[0])
                except Exception:
                    tf_dirs[tf_name]    = "?"
                    tf_returns[tf_name] = 0.0

            confidence = compute_confidence(direction, rsi14, adx_val, pdi, ndi,
                                            ema20_v, cur_px, tf_dirs)
            entry, entry_type = compute_entry(direction, cur_px, ema20_v)
            sl, tp, sl_dist   = compute_sl_tp(df, direction, entry, atr14,
                                               swing_bars, atr_mult, rr_ratio)
            risk_usd   = account * risk_pct
            pos_oz     = round(risk_usd / sl_dist, 4) if sl_dist > 0 else 0
            reward_usd = round(pos_oz * sl_dist * rr_ratio, 2)
            rr_actual  = abs(tp - entry) / sl_dist if sl_dist > 0 else 0

            # Store everything in session_state so reruns don't wipe it
            st.session_state.signal_data = dict(
                df=df, direction=direction, entry_tf=entry_tf,
                entry_type=entry_type, confidence=confidence,
                cur_px=cur_px, atr14=atr14, ema20_v=ema20_v, ema50_v=ema50_v,
                rsi14=rsi14, adx_val=adx_val, pdi=pdi, ndi=ndi,
                theta=theta, ets=ets, ens=ens, pred_return=pred_return,
                tf_dirs=tf_dirs, entry=entry, sl=sl, tp=tp,
                sl_dist=sl_dist, risk_usd=risk_usd, pos_oz=pos_oz,
                reward_usd=reward_usd, rr_actual=rr_actual,
                generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            )

    # Display from session_state (survives auto-reruns from live chart)
    if "signal_data" not in st.session_state:
        st.info("Configure settings in the sidebar, then click **Generate Signal**.")
        st.markdown("""
**How it works:**
1. Fetches live Gold (GC=F) data across all timeframes (1m → Daily)
2. Runs AutoTheta + AutoETS ensemble weighted by benchmark walk-forward Sharpe
3. Scores confidence from 5 factors: TF alignment, ADX, RSI, EMA position, DI cross
4. Calculates Entry (limit or market), Stop Loss (ATR + swing), Take Profit
5. Sizes position using fixed-fractional risk management
        """)
    else:
        s = st.session_state.signal_data
        direction  = s["direction"]
        arrow      = "▼" if direction == "SELL" else "▲"
        sig_color  = "green" if direction == "BUY" else "red"

        st.caption(f"Generated at {s['generated_at']}  |  Click Generate Signal to refresh.")

        col_sig, col_conf, col_px, col_tf, col_ret = st.columns([3, 1, 1, 1, 1])
        with col_sig:
            st.markdown(
                f"<h2 style='color:{'#00C853' if direction=='BUY' else '#FF1744'};margin:0'>"
                f"{arrow} {direction} &nbsp; XAUUSD [{s['entry_tf']}]</h2>",
                unsafe_allow_html=True,
            )
            st.caption(s["entry_type"])
        with col_conf:
            st.metric("Confidence", f"{s['confidence']}%")
        with col_px:
            st.metric("Gold Price", f"${s['cur_px']:,.2f}")
        with col_tf:
            aligned = sum(1 for d in s["tf_dirs"].values() if d == direction)
            st.metric("TF Aligned", f"{aligned}/{len(s['tf_dirs'])}")
        with col_ret:
            pr = s.get("pred_return", 0.0)
            st.metric("Model Return", f"{pr*100:+.3f}%",
                      help="Raw ensemble predicted return for bar+1. Below ±0.02% = noise.")

        st.divider()

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("ATR(14)",  f"${s['atr14']:.2f}")
        c2.metric("RSI(14)",  f"{s['rsi14']:.1f}")
        c3.metric("ADX(14)",  f"{s['adx_val']:.1f}", delta="Trending" if s['adx_val'] > 25 else "Ranging")
        c4.metric("EMA20",    f"${s['ema20_v']:,.2f}", delta="Above" if s['cur_px'] > s['ema20_v'] else "Below")
        c5.metric("EMA50",    f"${s['ema50_v']:,.2f}", delta="Above" if s['cur_px'] > s['ema50_v'] else "Below")

        st.divider()

        st.subheader("Multi-Timeframe Alignment")
        tf_cols = st.columns(len(s["tf_dirs"]))
        for i, (tf_name, d) in enumerate(s["tf_dirs"].items()):
            with tf_cols[i]:
                icon  = "▲" if d == "BUY" else ("▼" if d == "SELL" else "?")
                match = d == direction
                st.metric(tf_name, f"{icon} {d}",
                    delta="Aligned" if match else "Conflict",
                    delta_color="normal" if match else "inverse")

        st.divider()

        left, right = st.columns(2)
        with left:
            st.subheader("Ensemble Forecast")
            rows = []
            for h in [1, 2, 4]:
                cum_ens = float(np.cumsum(s["ens"])[h - 1])
                rows.append({
                    "Horizon":      f"+{h} bar{'s' if h > 1 else ''}",
                    "AutoTheta":    f"{np.cumsum(s['theta'])[h-1]*100:+.4f}%",
                    "AutoETS":      f"{np.cumsum(s['ets'])[h-1]*100:+.4f}%",
                    "Ensemble":     f"{cum_ens*100:+.4f}%",
                    "Price Target": f"${s['cur_px'] * np.exp(cum_ens):,.2f}",
                    "Direction":    "▲ UP" if cum_ens > 0 else "▼ DOWN",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        with right:
            st.subheader("Trade Plan")
            plan_df = pd.DataFrame([
                {"Field": "Direction",     "Value": f"{arrow} {direction}"},
                {"Field": "Entry",         "Value": f"${s['entry']:,.3f}"},
                {"Field": "Stop Loss",     "Value": f"${s['sl']:,.3f}  ({s['sl_dist']:.2f} pts = {atr_mult}x ATR)"},
                {"Field": "Take Profit",   "Value": f"${s['tp']:,.3f}  ({abs(s['tp']-s['entry']):.2f} pts)"},
                {"Field": "R:R",           "Value": f"1:{s['rr_actual']:.2f}"},
                {"Field": "Position Size", "Value": f"{s['pos_oz']:.4f} oz"},
                {"Field": "Risk $",        "Value": f"${s['risk_usd']:.2f}  ({risk_pct:.1%})"},
                {"Field": "Max Gain",      "Value": f"${s['reward_usd']:.2f}"},
            ])
            st.dataframe(plan_df, use_container_width=True, hide_index=True)

        st.divider()
        if s["confidence"] >= 70:
            st.success(f"Strong signal — {s['confidence']}% confidence. Key factors aligned.")
        elif s["confidence"] >= 55:
            st.warning(f"Moderate signal — {s['confidence']}% confidence. Consider 50% position size.")
        else:
            st.error(f"Weak signal — {s['confidence']}%. Timeframes conflicting. Best to skip.")

        if direction == "SELL" and s["rsi14"] < 35:
            st.warning("RSI oversold — counter-trend SELL. Higher reversal risk.")
        elif direction == "BUY" and s["rsi14"] > 65:
            st.warning("RSI overbought — counter-trend BUY. Higher reversal risk.")

        st.divider()
        st.subheader("Price Chart")
        df       = s["df"]
        ema20_s  = compute_ema(df["close"], 20)
        ema50_s  = compute_ema(df["close"], 50)
        rsi_s    = compute_rsi(df["close"], 14)
        fig = build_price_chart(df, ema20_s, ema50_s, rsi_s,
                                entry=s["entry"], sl=s["sl"], tp=s["tp"],
                                title=f"XAUUSD — {s['entry_tf']}")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"Last bar: {df.index[-1].strftime('%Y-%m-%d %H:%M UTC')}  |  "
            f"Models: AutoTheta + AutoETS (Sharpe-weighted)  |  Data: yfinance GC=F"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Live 1-Min Gold Chart
# ═══════════════════════════════════════════════════════════════════════════════

with tab_live:

    if "live_last" not in st.session_state:
        st.session_state.live_last = 0.0
    if "live_running" not in st.session_state:
        st.session_state.live_running = True

    col_hdr, col_btn = st.columns([5, 1])
    with col_hdr:
        st.markdown("### 📡 Live Gold — 1 Min")
    with col_btn:
        label = "⏸ Pause" if st.session_state.live_running else "▶ Resume"
        if st.button(label, use_container_width=True):
            st.session_state.live_running = not st.session_state.live_running

    chart_placeholder    = st.empty()
    stats_placeholder    = st.empty()
    progress_placeholder = st.empty()

    try:
        df_1m = fetch_data("1m", "1d")

        cur   = float(df_1m["close"].iloc[-1])
        prev  = float(df_1m["close"].iloc[-2])
        chg   = cur - prev
        pct   = chg / prev * 100
        high  = float(df_1m["high"].max())
        low   = float(df_1m["low"].min())
        atr1m = float(compute_atr(df_1m, 14).iloc[-1])

        with stats_placeholder.container():
            sc1, sc2, sc3, sc4, sc5 = st.columns(5)
            sc1.metric("Gold Price",   f"${cur:,.2f}", delta=f"{chg:+.2f} ({pct:+.2f}%)",
                       delta_color="normal" if chg >= 0 else "inverse")
            sc2.metric("Session High", f"${high:,.2f}")
            sc3.metric("Session Low",  f"${low:,.2f}")
            sc4.metric("ATR(14) 1m",  f"${atr1m:.3f}")
            sc5.metric("Bars",         f"{len(df_1m)}")

        ema20_1m = compute_ema(df_1m["close"], 20)
        ema50_1m = compute_ema(df_1m["close"], 50)
        rsi_1m   = compute_rsi(df_1m["close"], 14)
        fig_live = build_price_chart(
            df_1m, ema20_1m, ema50_1m, rsi_1m,
            title="XAUUSD — 1 Min (Live)", lookback=200,
        )
        fig_live.update_layout(height=660)

        with chart_placeholder.container():
            st.plotly_chart(fig_live, use_container_width=True)
            st.caption(
                f"Last bar: {df_1m.index[-1].strftime('%Y-%m-%d %H:%M:%S UTC')}  |  "
                f"Next refresh in {live_refresh}s  |  Source: yfinance GC=F"
            )

    except Exception as e:
        chart_placeholder.error(f"Failed to load 1-min data: {e}  —  Markets may be closed.")

    # Auto-refresh
    if st.session_state.live_running:
        now     = time.time()
        elapsed = now - st.session_state.live_last

        if elapsed >= live_refresh:
            st.session_state.live_last = now
            st.cache_data.clear()
            st.rerun()
        else:
            remaining = live_refresh - elapsed
            with progress_placeholder.container():
                st.progress(
                    1.0 - remaining / live_refresh,
                    text=f"Auto-refreshing in {int(remaining)}s  (interval: {live_refresh}s)",
                )
            time.sleep(min(remaining, 2.0))
            st.rerun()
    else:
        progress_placeholder.info("Live chart paused. Click **Resume** to restart auto-refresh.")
