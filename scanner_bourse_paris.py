#!/usr/bin/env python3
"""
Scanner Bourse de Paris — SBF120 / CAC All-Tradable
=====================================================

Ce script :
  1. Telecharge l'historique (yfinance) d'une liste d'actions parisiennes
  2. Calcule les indicateurs techniques (EMA20/50/200, RSI14, ATR14, volume moyen 20j, 52 semaines)
  3. Detecte les signaux : cassure 52 sem., pic de volume, croisement EMA20/EMA50,
     proximite EMA200 (+/-2%), volatilite ATR
  4. Calcule un score "momentum" (0-100) et classe chaque action dans 1 des 4 phases de cycle
  5. Detecte et memorise les "zones d'interet" (pics de volume marquants) -> memoire glissante
     de 20 zones max par action, stockee dans zones_memory.json (persistant entre les runs)
  6. Exporte un fichier JSON (scan_YYYY-MM-DD.json) pret a etre charge dans le tableau de bord

Usage :
    pip install yfinance pandas numpy --break-system-packages
    python scanner_bourse_paris.py

    Options utiles :
    python scanner_bourse_paris.py --tickers tickers_sbf120.csv --out scan_du_jour.json
    python scanner_bourse_paris.py --volume-multiple 2.0   # seuil pic de volume plus strict

A lancer chaque soir apres cloture (~17h35, la Bourse de Paris ferme a 17h30).
Le fichier JSON genere est ensuite importe dans le tableau de bord (dashboard_scanner.html).
"""

import argparse
import csv
import json
import os
import sys
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except ImportError:
    sys.exit(
        "Le module yfinance n'est pas installe.\n"
        "Lance : pip install yfinance pandas numpy --break-system-packages"
    )

# --------------------------------------------------------------------------------------
# Parametres par defaut (modifiables en ligne de commande)
# --------------------------------------------------------------------------------------
DEFAULT_TICKERS_CSV = "tickers_sbf120.csv"
DEFAULT_ZONES_FILE = "zones_memory.json"
DEFAULT_HISTORY = "18mo"      # profondeur d'historique telechargee (assez pour EMA200 + 52 sem.)
DEFAULT_VOLUME_MULTIPLE = 1.5  # seuil "volume superieur a la moyenne" pour le signal
ZONE_VOLUME_MULTIPLE = 2.0     # seuil pour qu'un jour devienne une "zone memoire" (pic de volume)
MAX_ZONES_PER_TICKER = 20
NEAR_EMA200_PCT = 0.02         # +/-2%


# --------------------------------------------------------------------------------------
# Indicateurs techniques
# --------------------------------------------------------------------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


# --------------------------------------------------------------------------------------
# Score momentum (0-100) + phase de cycle
# --------------------------------------------------------------------------------------
def compute_momentum_score(row) -> float:
    score = 0.0

    # 1) Alignement des moyennes mobiles (structure de tendance) -> 25 pts
    if row["ema20"] > row["ema50"] > row["ema200"]:
        score += 25
    elif row["ema20"] > row["ema50"]:
        score += 12
    elif row["close"] > row["ema200"]:
        score += 6

    # 2) Prix vs EMA20 -> 15 pts
    if row["close"] > row["ema20"]:
        score += 15

    # 3) RSI -> 20 pts (zone saine 50-70), penalise si surachat ou survente
    r = row["rsi14"]
    if 50 <= r <= 70:
        score += 20
    elif 40 <= r < 50:
        score += 10
    elif 70 < r <= 80:
        score += 8
    elif r > 80:
        score += 2

    # 4) Confirmation de volume -> 15 pts
    if row["vol_ratio"] >= 1.5:
        score += 15
    elif row["vol_ratio"] >= 1.0:
        score += 7

    # 5) Distance au plus haut 52 semaines -> 15 pts (proche = force relative)
    d = row["dist_52w_pct"]  # negatif ou nul : 0 = au plus haut
    if d >= -3:
        score += 15
    elif d >= -8:
        score += 10
    elif d >= -15:
        score += 5

    # 6) Volatilite ATR en expansion saine -> 10 pts
    if row["atr_rising"] and row["close"] > row["ema50"]:
        score += 10
    elif row["atr_rising"]:
        score += 4

    return round(min(score, 100), 1)


def classify_phase(row) -> str:
    """Classe l'action dans un des 4 quadrants du cycle de tendance."""
    close, ema20, ema50, ema200 = row["close"], row["ema20"], row["ema50"], row["ema200"]
    r = row["rsi14"]
    bullish_stack = ema20 > ema50 > ema200
    ext_pct = (close - ema20) / ema20 * 100 if ema20 else 0

    # Phase 4 - Distribution / faiblesse : la tendance haussiere se casse
    if (row["ema_cross"] == "death") or (ema20 < ema50 and close < ema20) or (
        r < 60 and row.get("rsi_falling_from_high", False)
    ):
        return "distribution"

    # Phase 3 - Extension : mouvement trop avance, risque de correction
    if r > 70 and (ext_pct > 8 or row["vol_ratio"] > 2.5):
        return "extension"
    if bullish_stack and ext_pct > 12:
        return "extension"

    # Phase 2 - Momentum haussier : zone ideale
    if bullish_stack and 50 <= r <= 75 and close > ema20:
        return "momentum"

    # Phase 1 - Accumulation : le marche se reveille
    if close > ema200 and not bullish_stack and 38 <= r <= 62:
        return "accumulation"

    # Cas transitoires : on rattache au plus proche par un score directionnel
    if close < ema200:
        return "distribution" if r < 45 else "accumulation"
    return "accumulation"


PHASE_LABELS = {
    "accumulation": "Phase 1 - Accumulation",
    "momentum": "Phase 2 - Momentum haussier",
    "extension": "Phase 3 - Extension",
    "distribution": "Phase 4 - Distribution / faiblesse",
}


# --------------------------------------------------------------------------------------
# Zones memoire (pics de volume marquants)
# --------------------------------------------------------------------------------------
def load_zones_memory(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_zones_memory(path: str, memory: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)


def update_zones_for_ticker(memory: dict, ticker: str, df: pd.DataFrame) -> list:
    """Detecte les pics de volume (>= ZONE_VOLUME_MULTIPLE x moyenne 20j) sur tout
    l'historique telecharge, fusionne avec les zones deja memorisees, et ne garde
    que les MAX_ZONES_PER_TICKER plus recentes (par date)."""
    existing = {z["date"]: z for z in memory.get(ticker, [])}

    candidates = df[df["vol_ratio"] >= ZONE_VOLUME_MULTIPLE]
    for date, r in candidates.iterrows():
        date_str = date.strftime("%Y-%m-%d")
        existing[date_str] = {
            "date": date_str,
            "low": round(float(r["Low"]), 3),
            "high": round(float(r["High"]), 3),
            "close": round(float(r["Close"]), 3),
            "vol_ratio": round(float(r["vol_ratio"]), 2),
        }

    all_zones = sorted(existing.values(), key=lambda z: z["date"], reverse=True)
    trimmed = all_zones[:MAX_ZONES_PER_TICKER]
    memory[ticker] = trimmed
    return trimmed


def closest_zones_to_price(zones: list, price: float, n: int = 3) -> list:
    if not price or not zones:
        return []
    ranked = sorted(zones, key=lambda z: abs(((z["low"] + z["high"]) / 2) - price))
    return ranked[:n]


# --------------------------------------------------------------------------------------
# Pipeline principal
# --------------------------------------------------------------------------------------
def load_tickers(csv_path: str):
    tickers, names = [], {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t = row["ticker"].strip()
            tickers.append(t)
            names[t] = row["nom"].strip()
    return tickers, names


def process_ticker(ticker: str, name: str, raw: pd.DataFrame, memory: dict, volume_multiple: float):
    if raw is None or raw.empty or len(raw) < 210:
        return None, f"{ticker}: historique insuffisant ({0 if raw is None else len(raw)} lignes) - ignore"

    df = raw.copy()
    df = df.dropna(subset=["Close", "Volume"])
    df["ema20"] = ema(df["Close"], 20)
    df["ema50"] = ema(df["Close"], 50)
    df["ema200"] = ema(df["Close"], 200)
    df["rsi14"] = rsi(df["Close"], 14)
    df["atr14"] = atr(df, 14)
    df["avg_vol20"] = df["Volume"].rolling(20).mean()
    df["vol_ratio"] = df["Volume"] / df["avg_vol20"].replace(0, np.nan)
    df["high_52w"] = df["High"].rolling(252, min_periods=60).max()
    df["ema_diff"] = df["ema20"] - df["ema50"]
    df["atr_rising"] = df["atr14"] > df["atr14"].shift(14)

    if len(df) < 60:
        return None, f"{ticker}: pas assez de donnees apres calcul - ignore"

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # Croisement EMA20/EMA50 (detecte sur le dernier jour)
    ema_cross = None
    if prev["ema_diff"] <= 0 < last["ema_diff"]:
        ema_cross = "golden"
    elif prev["ema_diff"] >= 0 > last["ema_diff"]:
        ema_cross = "death"

    # Cassure des plus hauts 52 semaines (le close du jour depasse le plus haut des 252
    # jours precedents, hors jour courant)
    prior_high_52w = df["High"].shift(1).rolling(252, min_periods=60).max().iloc[-1]
    breakout_52w = bool(last["Close"] >= prior_high_52w) if pd.notna(prior_high_52w) else False

    dist_52w_pct = (
        round((last["Close"] - last["high_52w"]) / last["high_52w"] * 100, 2)
        if pd.notna(last["high_52w"]) and last["high_52w"] else 0.0
    )

    near_ema200 = bool(abs(last["Close"] - last["ema200"]) / last["ema200"] <= NEAR_EMA200_PCT) if last["ema200"] else False

    vol_ratio = float(last["vol_ratio"]) if pd.notna(last["vol_ratio"]) else 0.0
    volume_spike = vol_ratio >= volume_multiple

    # RSI en repli depuis une zone de surachat (utile pour la phase 4)
    rsi_recent_max = df["rsi14"].tail(10).max()
    rsi_falling_from_high = bool(rsi_recent_max > 70 and last["rsi14"] < rsi_recent_max - 8)

    row = {
        "close": float(last["Close"]),
        "ema20": float(last["ema20"]),
        "ema50": float(last["ema50"]),
        "ema200": float(last["ema200"]),
        "rsi14": float(last["rsi14"]),
        "atr14": float(last["atr14"]),
        "atr_pct": round(float(last["atr14"]) / float(last["Close"]) * 100, 2) if last["Close"] else 0,
        "atr_rising": bool(last["atr_rising"]) if pd.notna(last["atr_rising"]) else False,
        "vol_ratio": vol_ratio,
        "dist_52w_pct": dist_52w_pct,
        "ema_cross": ema_cross,
        "rsi_falling_from_high": rsi_falling_from_high,
    }

    momentum_score = compute_momentum_score(row)
    phase = classify_phase(row)

    zones = update_zones_for_ticker(memory, ticker, df)
    top_zones = closest_zones_to_price(zones, row["close"], n=3)

    result = {
        "ticker": ticker,
        "nom": name,
        "date": df.index[-1].strftime("%Y-%m-%d"),
        "close": round(row["close"], 3),
        "ema20": round(row["ema20"], 3),
        "ema50": round(row["ema50"], 3),
        "ema200": round(row["ema200"], 3),
        "rsi14": round(row["rsi14"], 1),
        "atr14": round(row["atr14"], 3),
        "atr_pct": row["atr_pct"],
        "atr_rising": row["atr_rising"],
        "vol_ratio": round(vol_ratio, 2),
        "high_52w": round(float(last["high_52w"]), 3) if pd.notna(last["high_52w"]) else None,
        "dist_52w_pct": dist_52w_pct,
        "signals": {
            "breakout_52w": breakout_52w,
            "volume_spike": volume_spike,
            "ema_cross": ema_cross,
            "near_ema200": near_ema200,
        },
        "momentum_score": momentum_score,
        "phase": phase,
        "phase_label": PHASE_LABELS[phase],
        "zones_top3": top_zones,
        "zones_count": len(zones),
    }
    return result, None


def main():
    parser = argparse.ArgumentParser(description="Scanner technique - Bourse de Paris")
    parser.add_argument("--tickers", default=DEFAULT_TICKERS_CSV, help="CSV ticker,nom")
    parser.add_argument("--out", default=None, help="Fichier JSON de sortie")
    parser.add_argument("--zones-file", default=DEFAULT_ZONES_FILE, help="Fichier memoire des zones (persistant)")
    parser.add_argument("--history", default=DEFAULT_HISTORY, help="Profondeur d'historique yfinance (ex: 18mo, 2y)")
    parser.add_argument("--volume-multiple", type=float, default=DEFAULT_VOLUME_MULTIPLE,
                         help="Seuil de declenchement du signal 'volume superieur a la moyenne'")
    args = parser.parse_args()

    today_str = datetime.now().strftime("%Y-%m-%d")
    out_path = args.out or f"scan_{today_str}.json"

    tickers, names = load_tickers(args.tickers)
    print(f"[*] {len(tickers)} tickers charges depuis {args.tickers}")

    print(f"[*] Telechargement des donnees ({args.history})...")
    raw_data = yf.download(
        tickers, period=args.history, group_by="ticker", auto_adjust=True,
        threads=True, progress=False,
    )

    memory = load_zones_memory(args.zones_file)

    results, errors = [], []
    for t in tickers:
        try:
            df_t = raw_data[t] if isinstance(raw_data.columns, pd.MultiIndex) else raw_data
        except KeyError:
            errors.append(f"{t}: donnees introuvables (ticker invalide ?)")
            continue

        res, err = process_ticker(t, names.get(t, t), df_t, memory, args.volume_multiple)
        if res:
            results.append(res)
        if err:
            errors.append(err)

    save_zones_memory(args.zones_file, memory)

    results.sort(key=lambda r: r["momentum_score"], reverse=True)

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scan_date": today_str,
        "nb_tickers_ok": len(results),
        "nb_tickers_erreur": len(errors),
        "tickers": results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"[OK] {len(results)} actions analysees -> {out_path}")
    print(f"[OK] Memoire des zones mise a jour -> {args.zones_file}")
    if errors:
        print(f"[!] {len(errors)} tickers en erreur (verifie les symboles dans {args.tickers}) :")
        for e in errors[:15]:
            print("    -", e)
        if len(errors) > 15:
            print(f"    ... et {len(errors) - 15} de plus")

    # Repartition par phase, pour un apercu rapide en console
    from collections import Counter
    counts = Counter(r["phase"] for r in results)
    print("\nRepartition par phase :")
    for p in ["accumulation", "momentum", "extension", "distribution"]:
        print(f"  {PHASE_LABELS[p]:<38} {counts.get(p, 0)}")


if __name__ == "__main__":
    main()
