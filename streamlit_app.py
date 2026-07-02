"""
Scanner Bourse de Paris — application Streamlit
==================================================
Lit le fichier scan_du_jour.json (genere par scanner_bourse_paris.py, mis a jour
automatiquement chaque soir par le workflow GitHub Actions) et l'affiche sous
forme de tableau de bord interactif.

Lancement local :
    pip install -r requirements.txt
    streamlit run streamlit_app.py

Deploiement : Streamlit Community Cloud, en pointant sur ce fichier.
"""

import json
import os
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Scanner Bourse de Paris", layout="wide", page_icon="📈")

SCAN_FILE = "scan_du_jour.json"

PHASES = {
    "accumulation": {"label": "Phase 1 · Accumulation", "color": "#4C8DFF", "angle_start": 270},
    "momentum": {"label": "Phase 2 · Momentum haussier", "color": "#2ECC8F", "angle_start": 0},
    "extension": {"label": "Phase 3 · Extension", "color": "#F5A623", "angle_start": 90},
    "distribution": {"label": "Phase 4 · Distribution / faiblesse", "color": "#FF5C5C", "angle_start": 180},
}

# --------------------------------------------------------------------------------------
# Styles
# --------------------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main {background-color:#0E1116;}
    h1, h2, h3 {font-family: 'Georgia', serif;}
    .stMetric {background:#161B22; padding:10px 14px; border-radius:10px; border:1px solid #262D38;}
    div[data-testid="stDataFrame"] {border:1px solid #262D38; border-radius:10px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------------------
# Chargement des donnees
# --------------------------------------------------------------------------------------
@st.cache_data(ttl=300)
def load_scan(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


uploaded = st.sidebar.file_uploader("Importer un scan JSON (optionnel)", type="json")
if uploaded:
    data = json.load(uploaded)
else:
    data = load_scan(SCAN_FILE)

st.title("Scanner Bourse de Paris")
st.caption("SBF 120 / CAC All-Tradable — cassures, volumes, EMA, cycle de tendance")

if data is None:
    st.info(
        f"Aucun fichier `{SCAN_FILE}` trouve. Lance `scanner_bourse_paris.py` (localement ou "
        f"via le workflow GitHub Actions) ou importe un JSON depuis la barre laterale."
    )
    st.stop()

df = pd.DataFrame(data["tickers"])
df["ema_cross_flag"] = df["signals"].apply(lambda s: s.get("ema_cross"))
df["breakout_52w"] = df["signals"].apply(lambda s: s.get("breakout_52w"))
df["volume_spike"] = df["signals"].apply(lambda s: s.get("volume_spike"))
df["near_ema200"] = df["signals"].apply(lambda s: s.get("near_ema200"))

gen_dt = datetime.fromisoformat(data["generated_at"].replace("Z", "+00:00"))
st.caption(
    f"Scan du **{data['scan_date']}** · {data['nb_tickers_ok']} valeurs analysees "
    f"· genere a {gen_dt.strftime('%H:%M')} UTC"
)

# --------------------------------------------------------------------------------------
# Barre laterale : filtres
# --------------------------------------------------------------------------------------
st.sidebar.header("Filtres")
phase_filter = st.sidebar.multiselect(
    "Phase", options=list(PHASES.keys()), default=list(PHASES.keys()),
    format_func=lambda k: PHASES[k]["label"],
)
sig_breakout = st.sidebar.checkbox("Cassure 52 semaines")
sig_volume = st.sidebar.checkbox("Pic de volume")
sig_cross = st.sidebar.checkbox("Croisement EMA20/50 (golden)")
sig_ema200 = st.sidebar.checkbox("Proche EMA200 (+/-2%)")
search = st.sidebar.text_input("Recherche (ticker ou nom)")

filtered = df[df["phase"].isin(phase_filter)]
if sig_breakout:
    filtered = filtered[filtered["breakout_52w"]]
if sig_volume:
    filtered = filtered[filtered["volume_spike"]]
if sig_cross:
    filtered = filtered[filtered["ema_cross_flag"] == "golden"]
if sig_ema200:
    filtered = filtered[filtered["near_ema200"]]
if search:
    q = search.lower()
    filtered = filtered[
        filtered["ticker"].str.lower().str.contains(q) | filtered["nom"].str.lower().str.contains(q)
    ]

# --------------------------------------------------------------------------------------
# Repartition par phase (metrics)
# --------------------------------------------------------------------------------------
cols = st.columns(4)
counts = df["phase"].value_counts()
for col, (key, meta) in zip(cols, PHASES.items()):
    col.metric(meta["label"], int(counts.get(key, 0)))

# --------------------------------------------------------------------------------------
# Horloge des cycles (scatter polaire)
# --------------------------------------------------------------------------------------
left, right = st.columns([1, 1.6])

with left:
    st.subheader("Horloge des cycles")

    import hashlib

    def angle_offset(ticker, spread=84):
        h = int(hashlib.md5(ticker.encode()).hexdigest(), 16)
        return (h % 1000) / 1000 * spread + 3

    xs, ys, colors, texts = [], [], [], []
    for _, row in df.iterrows():
        meta = PHASES.get(row["phase"])
        if not meta:
            continue
        angle_deg = meta["angle_start"] + angle_offset(row["ticker"])
        import math

        rad = math.radians(angle_deg)
        r = 15 + (row["momentum_score"] / 100) * 90
        xs.append(r * math.sin(rad))
        ys.append(-r * math.cos(rad))
        colors.append(meta["color"])
        texts.append(f"{row['ticker']} — {row['nom']}<br>Score {row['momentum_score']} — {meta['label']}")

    fig = go.Figure()
    fig.add_shape(type="line", x0=-115, y0=0, x1=115, y1=0, line=dict(color="#262D38", width=1))
    fig.add_shape(type="line", x0=0, y0=-115, x1=0, y1=115, line=dict(color="#262D38", width=1))
    fig.add_shape(
        type="circle", x0=-105, y0=-105, x1=105, y1=105,
        line=dict(color="#262D38", width=1),
    )
    fig.add_trace(
        go.Scatter(
            x=xs, y=ys, mode="markers",
            marker=dict(color=colors, size=9, opacity=0.85, line=dict(width=0)),
            text=texts, hoverinfo="text",
        )
    )
    annotations = [
        dict(x=-80, y=45, text="Accumulation", showarrow=False, font=dict(color="#8792A2", size=11)),
        dict(x=80, y=100, text="Momentum", showarrow=False, font=dict(color="#8792A2", size=11)),
        dict(x=90, y=-80, text="Extension", showarrow=False, font=dict(color="#8792A2", size=11)),
        dict(x=-70, y=-100, text="Distribution", showarrow=False, font=dict(color="#8792A2", size=11)),
    ]
    fig.update_layout(
        plot_bgcolor="#161B22", paper_bgcolor="#161B22",
        xaxis=dict(visible=False, range=[-130, 130]),
        yaxis=dict(visible=False, range=[-130, 130], scaleanchor="x"),
        annotations=annotations, height=430, margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------------------
# Tableau
# --------------------------------------------------------------------------------------
with right:
    st.subheader(f"Valeurs ({len(filtered)})")
    table = filtered[[
        "ticker", "nom", "phase", "momentum_score", "close", "dist_52w_pct",
        "rsi14", "vol_ratio", "atr_pct",
    ]].copy()
    table["phase"] = table["phase"].map(lambda k: PHASES[k]["label"])
    table.columns = ["Ticker", "Nom", "Phase", "Score", "Prix (EUR)", "% / 52 sem.", "RSI14", "Vol/moy20", "ATR %"]
    table = table.sort_values("Score", ascending=False)
    st.dataframe(table, use_container_width=True, height=430, hide_index=True)

# --------------------------------------------------------------------------------------
# Detail d'une valeur + zones memoire
# --------------------------------------------------------------------------------------
st.divider()
st.subheader("Detail d'une valeur")

ticker_options = filtered["ticker"].tolist() if len(filtered) else df["ticker"].tolist()
if ticker_options:
    chosen = st.selectbox(
        "Choisir une valeur",
        options=ticker_options,
        format_func=lambda t: f"{t} — {df[df['ticker']==t]['nom'].values[0]}",
    )
    row = df[df["ticker"] == chosen].iloc[0]
    meta = PHASES[row["phase"]]

    st.markdown(
        f"### {row['ticker']} — {row['nom']}  "
        f":{'blue' if row['phase']=='accumulation' else 'green' if row['phase']=='momentum' else 'orange' if row['phase']=='extension' else 'red'}[{meta['label']}]"
    )

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Prix", f"{row['close']:.2f} €")
    k2.metric("EMA20", f"{row['ema20']:.2f}")
    k3.metric("EMA50", f"{row['ema50']:.2f}")
    k4.metric("EMA200", f"{row['ema200']:.2f}")
    k5.metric("RSI14", f"{row['rsi14']:.1f}")
    k6.metric("ATR14", f"{row['atr14']:.2f} ({row['atr_pct']}%)")

    st.markdown("**Zones memoire les plus proches du cours actuel**")
    zones = row.get("zones_top3") or []
    if zones:
        zdf = pd.DataFrame(zones)
        zdf["milieu"] = (zdf["low"] + zdf["high"]) / 2
        zdf["distance_%"] = ((zdf["milieu"] - row["close"]) / row["close"] * 100).round(1)
        zdf = zdf[["date", "low", "high", "distance_%", "vol_ratio"]]
        zdf.columns = ["Date", "Bas", "Haut", "Distance %", "Volume (x moy.)"]
        st.dataframe(zdf, use_container_width=True, hide_index=True)
    else:
        st.caption(
            "Aucune zone de volume marquant enregistree pour l'instant. "
            "Elles s'accumulent au fil des scans quotidiens (memoire glissante de 20 zones)."
        )
else:
    st.info("Aucune valeur ne correspond aux filtres selectionnes.")
