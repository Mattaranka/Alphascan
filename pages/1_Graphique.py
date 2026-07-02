"""
Page Graphique — Scanner Bourse de Paris
==========================================
Graphique en chandeliers pour une valeur, avec choix de la periode (1j a 1 an),
superposition des EMA20/50/200 et des zones de support/resistance (zones memoire
du scanner + niveaux calcules sur la fenetre affichee).
"""

import csv
import json
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Graphique — Scanner Bourse de Paris", layout="wide", page_icon="📈")

SCAN_FILE = "scan_du_jour.json"
ZONES_FILE = "zones_memory.json"
TICKERS_FILE = "tickers_sbf120.csv"

st.title("Graphique detaille")


# --------------------------------------------------------------------------------------
# Chargement de la liste des valeurs (depuis le dernier scan, sinon le CSV)
# --------------------------------------------------------------------------------------
def load_options():
    if os.path.exists(SCAN_FILE):
        data = json.load(open(SCAN_FILE, encoding="utf-8"))
        return [(t["ticker"], t["nom"]) for t in data["tickers"]]
    if os.path.exists(TICKERS_FILE):
        with open(TICKERS_FILE, newline="", encoding="utf-8") as f:
            return [(r["ticker"], r["nom"]) for r in csv.DictReader(f)]
    return []


options = load_options()
if not options:
    st.warning("Aucune liste de valeurs disponible (ni scan_du_jour.json, ni tickers_sbf120.csv).")
    st.stop()

tickers_only = [o[0] for o in options]
names = dict(options)

default_ticker = st.session_state.get("selected_ticker")
default_idx = tickers_only.index(default_ticker) if default_ticker in tickers_only else 0

col_a, col_b = st.columns([2, 3])
with col_a:
    chosen = st.selectbox(
        "Valeur", options=tickers_only, index=default_idx,
        format_func=lambda t: f"{t} — {names.get(t, t)}",
    )
with col_b:
    period_label = st.radio(
        "Periode", ["1 jour", "5 jours", "1 mois", "3 mois", "6 mois", "1 an"],
        horizontal=True, index=3,
    )

PERIOD_CONFIG = {
    "1 jour": {"period": "1d", "interval": "5m", "intraday": True},
    "5 jours": {"period": "5d", "interval": "15m", "intraday": True},
    "1 mois": {"days": 22, "intraday": False},
    "3 mois": {"days": 65, "intraday": False},
    "6 mois": {"days": 130, "intraday": False},
    "1 an": {"days": 260, "intraday": False},
}
cfg = PERIOD_CONFIG[period_label]


# --------------------------------------------------------------------------------------
# Donnees
# --------------------------------------------------------------------------------------
def flatten_cols(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


@st.cache_data(ttl=1800)
def fetch_daily_long(ticker):
    df = yf.download(ticker, period="2y", interval="1d", auto_adjust=True, progress=False)
    return flatten_cols(df)


@st.cache_data(ttl=300)
def fetch_intraday(ticker, period, interval):
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    return flatten_cols(df)


def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


daily = fetch_daily_long(chosen)
if daily.empty:
    st.error("Impossible de recuperer les donnees pour cette valeur (ticker invalide ou Yahoo Finance indisponible).")
    st.stop()

daily["ema20"] = ema(daily["Close"], 20)
daily["ema50"] = ema(daily["Close"], 50)
daily["ema200"] = ema(daily["Close"], 200)

if cfg["intraday"]:
    chart_df = fetch_intraday(chosen, cfg["period"], cfg["interval"])
    ema_mode = "flat"
else:
    chart_df = daily.tail(cfg["days"])
    ema_mode = "trail"

if chart_df.empty:
    st.error("Pas de donnees pour cette periode (marche ferme, ou intervalle indisponible pour cette valeur).")
    st.stop()


# --------------------------------------------------------------------------------------
# Zones memoire (support/resistance issues du scanner)
# --------------------------------------------------------------------------------------
def load_zones(ticker):
    if os.path.exists(ZONES_FILE):
        mem = json.load(open(ZONES_FILE, encoding="utf-8"))
        return mem.get(ticker, [])
    return []


def swing_levels(df, window=3, tol_pct=0.8, max_levels=4):
    """Detecte des niveaux de support/resistance a partir des points hauts/bas locaux
    de la fenetre affichee, puis regroupe les niveaux proches entre eux."""
    highs, lows = [], []
    h, l = df["High"].values, df["Low"].values
    for i in range(window, len(df) - window):
        if h[i] == max(h[i - window:i + window + 1]):
            highs.append(float(h[i]))
        if l[i] == min(l[i - window:i + window + 1]):
            lows.append(float(l[i]))

    def cluster(vals):
        vals = sorted(vals)
        clusters = []
        for v in vals:
            if clusters and abs(v - clusters[-1][-1]) / clusters[-1][-1] * 100 <= tol_pct:
                clusters[-1].append(v)
            else:
                clusters.append([v])
        return [sum(c) / len(c) for c in clusters]

    return cluster(lows)[:max_levels], cluster(highs)[-max_levels:]


# --------------------------------------------------------------------------------------
# Graphique
# --------------------------------------------------------------------------------------
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=chart_df.index, open=chart_df["Open"], high=chart_df["High"],
    low=chart_df["Low"], close=chart_df["Close"], name=chosen,
    increasing_line_color="#2ECC8F", decreasing_line_color="#FF5C5C",
))

if ema_mode == "trail":
    for col, color, label in [("ema20", "#4C8DFF", "EMA20"), ("ema50", "#F5A623", "EMA50"), ("ema200", "#B08D57", "EMA200")]:
        series = daily[col].reindex(chart_df.index)
        fig.add_trace(go.Scatter(x=chart_df.index, y=series, mode="lines", name=label, line=dict(width=1.3, color=color)))
else:
    last = daily.iloc[-1]
    for col, color, label in [("ema20", "#4C8DFF", "EMA20 (veille)"), ("ema50", "#F5A623", "EMA50 (veille)"), ("ema200", "#B08D57", "EMA200 (veille)")]:
        fig.add_hline(y=float(last[col]), line_dash="dot", line_color=color, opacity=0.75,
                       annotation_text=label, annotation_position="right")

# Zones memoire (pics de volume historiques) -> bandes horizontales
zones = load_zones(chosen)
visible_low, visible_high = float(chart_df["Low"].min()), float(chart_df["High"].max())
margin = (visible_high - visible_low) * 2 or visible_high * 0.05
for z in zones:
    mid = (z["low"] + z["high"]) / 2
    if visible_low - margin <= mid <= visible_high + margin:
        fig.add_hrect(y0=z["low"], y1=z["high"], fillcolor="#B08D57", opacity=0.16, line_width=0)
        fig.add_annotation(
            x=chart_df.index[-1], y=mid, xanchor="left", showarrow=False,
            text=f"  zone {z['date']} ({z['vol_ratio']}x vol.)", font=dict(size=9, color="#B08D57"),
        )

# Support / resistance calcules sur la fenetre affichee
if len(chart_df) > 10:
    sup, res = swing_levels(chart_df)
    for s in sup:
        fig.add_hline(y=s, line_dash="dash", line_color="#2ECC8F", opacity=0.45)
    for r in res:
        fig.add_hline(y=r, line_dash="dash", line_color="#FF5C5C", opacity=0.45)

fig.update_layout(
    height=620, plot_bgcolor="#161B22", paper_bgcolor="#161B22",
    font=dict(color="#E9EDF1"),
    xaxis_rangeslider_visible=False,
    legend=dict(orientation="h", y=1.06),
    margin=dict(l=10, r=10, t=40, b=10),
)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Bandes brunes = zones memoire (pics de volume detectes par le scanner quotidien). "
    "Lignes pointillees vertes = supports, rouges = resistances (calcules sur la fenetre affichee). "
    "Sur les vues 1 jour / 5 jours, les EMA sont affichees en reference plate (valeur de cloture de la veille), "
    "car elles sont calculees sur des clotures journalieres."
)
