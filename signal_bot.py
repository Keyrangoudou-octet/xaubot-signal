# ══════════════════════════════════════════════════════════════
# XauBot — Signal Bot v26 TENDANCE + LIQUIDITÉ + CASSURE
#
# Stratégie choisie par backtest sur 14 mois de bougies réelles Vantage (oct 2025 → nov 2026),
# vérifiée sur une période jamais vue (juil → nov 2026) :
#   1. TENDANCE   : clôture journalière d'hier au-dessus de son EMA20 → BUY seulement,
#                   en dessous → SELL seulement
#   2. LIQUIDITÉ  : range d'Asie = plus haut / plus bas de 00h00 à 07h00 UTC
#   3. CASSURE    : de 07h00 à 18h00 UTC, une bougie M5 CLÔTURE au-delà du range
#                   dans le sens de la tendance → « 🎯 ENTRE MAINTENANT »
#   SL = milieu du range | TP = 2 R | si rien n'est touché : on ferme à 21h00 UTC
#   1 seul trade par jour. Signaux uniquement : le bot ne passe AUCUN ordre.
# ══════════════════════════════════════════════════════════════

import asyncio
import logging
import os
import time
import httpx
import pandas as pd

from datetime import datetime, timezone
from telegram import Bot
from telegram.request import HTTPXRequest

TELEGRAM_TOKEN   = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
TWELVE_API_KEY   = os.environ["TWELVE_API_KEY"]

SYMBOL           = "XAU/USD"
REQUEST_TIMEOUT  = 15
SCAN_OFFSET_SEC  = 20          # scan 20 s après chaque clôture M5

# ── Stratégie (valeurs du backtest) — heures en UTC ─────────────
RANGE_END_MIN    = 7 * 60      # range d'Asie : 00h00 → 07h00 UTC
ENTRY_END_MIN    = 18 * 60     # cassures acceptées jusqu'à 18h00 UTC
CLOSE_MIN        = 21 * 60     # fermeture forcée à 21h00 UTC
DAILY_EMA        = 20
TP_R             = 2.0
MIN_RISK_USD     = 0.3
RISK_EUR         = float(os.environ.get("RISK_EUR", "100"))   # 1 % d'un compte de 10 000 €

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

state = {
    "day": None,        # jour UTC en cours
    "range": None,      # (haut, bas)
    "trend": None,      # 1 = BUY seulement, -1 = SELL seulement, 0 = rien
    "done": False,      # trade du jour déjà pris (ou journée terminée)
    "last_bar": None,   # dernière bougie M5 analysée
    "trade": None,
    "credits": 0, "credit_day": None,
}

# ══════════════════════════════════════════════════════════════
#  Horloge
# ══════════════════════════════════════════════════════════════
def now_utc():
    return datetime.now(timezone.utc)

def minute_of_day(dt):
    return dt.hour * 60 + dt.minute

def seconds_to_next_m5():
    now = time.time()
    return max(5, (int(now) // 300 + 1) * 300 + SCAN_OFFSET_SEC - now)

# ══════════════════════════════════════════════════════════════
#  Données Twelve Data (heures en UTC)
# ══════════════════════════════════════════════════════════════
async def get_candles(interval="5min", outputsize=120):
    state["credits"] += 1
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            r = await client.get("https://api.twelvedata.com/time_series",
                                 params={"symbol": SYMBOL, "interval": interval, "outputsize": outputsize,
                                         "timezone": "UTC", "apikey": TWELVE_API_KEY, "format": "JSON"})
            data = r.json()
        if "values" not in data:
            log.error("Twelve Data %s: %s", interval, data.get("message", data)); return None
        df = pd.DataFrame(data["values"]).rename(columns={"datetime": "time"})
        for c in ("open", "high", "low", "close"): df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.iloc[::-1].reset_index(drop=True).dropna(subset=["open", "high", "low", "close"])
        df["time"] = pd.to_datetime(df["time"])
        return df if len(df) >= 3 else None
    except Exception as e:
        log.error("get_candles %s: %s", interval, e); return None

async def get_live_price():
    state["credits"] += 1
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            r = await client.get("https://api.twelvedata.com/price", params={"symbol": SYMBOL, "apikey": TWELVE_API_KEY})
            return float(r.json()["price"])
    except Exception as e:
        log.error("get_live_price: %s", e); return None

def closed(df):
    return None if df is None or len(df) < 2 else df.iloc[:-1].copy()     # dernière bougie = en formation

# ══════════════════════════════════════════════════════════════
#  Logique (fonctions pures → testables)
# ══════════════════════════════════════════════════════════════
def daily_trend(df_day):
    """df_day = bougies journalières FERMÉES. 1 si la dernière clôture > EMA20, -1 si <, 0 sinon."""
    if df_day is None or len(df_day) < DAILY_EMA: return 0
    c = df_day["close"]; e = c.ewm(span=DAILY_EMA, adjust=False).mean()
    diff = c.iloc[-1] - e.iloc[-1]
    return 1 if diff > 0 else (-1 if diff < 0 else 0)

def asia_range(df5, day):
    """Plus haut / plus bas des bougies M5 du jour qui commencent avant 07h00 UTC."""
    if df5 is None: return None
    t = df5["time"]
    m = (t.dt.normalize() == pd.Timestamp(day)) & ((t.dt.hour * 60 + t.dt.minute) < RANGE_END_MIN)
    if m.sum() < 3: return None
    return float(df5.loc[m, "high"].max()), float(df5.loc[m, "low"].min())

def breakout(close, rng, trend):
    """Retourne 1 (BUY) / -1 (SELL) si la clôture casse le range dans le sens de la tendance, sinon 0."""
    hi, lo = rng
    if close > hi and trend > 0: return 1
    if close < lo and trend < 0: return -1
    return 0

def build_trade(s, entry, rng, bar_time):
    hi, lo = rng
    sl = (hi + lo) / 2; risk = abs(entry - sl)
    if risk <= MIN_RISK_USD or (s > 0 and sl >= entry) or (s < 0 and sl <= entry): return None
    return {"dir": "BUY" if s > 0 else "SELL", "s": s, "entry": round(entry, 2), "sl": round(sl, 2),
            "tp": round(entry + s * TP_R * risk, 2), "risk": round(risk, 2),
            "first_bar": pd.Timestamp(bar_time) + pd.Timedelta(minutes=5), "opened": now_utc()}

def check_exit(t, bars):
    """bars = bougies M5 fermées. On regarde celles d'après la bougie de cassure. SL prioritaire."""
    s = t["s"]
    for _, b in bars[bars["time"] >= t["first_bar"]].iterrows():
        if (s > 0 and b["low"] <= t["sl"]) or (s < 0 and b["high"] >= t["sl"]): return "SL", -1.0
        if (s > 0 and b["high"] >= t["tp"]) or (s < 0 and b["low"] <= t["tp"]): return "TP", TP_R
    return None, None

def lot_for(risk_usd):
    if risk_usd <= 0: return 0.01
    return max(0.01, int(RISK_EUR / (risk_usd * 100) * 100) / 100)

# ══════════════════════════════════════════════════════════════
#  Messages
# ══════════════════════════════════════════════════════════════
def format_range(rng, trend, day):
    hi, lo = rng
    if trend > 0:   plan = f"📈 Tendance du jour HAUSSIÈRE → BUY seulement\n🎯 J'attends une clôture M5 AU-DESSUS de {hi}"
    elif trend < 0: plan = f"📉 Tendance du jour BAISSIÈRE → SELL seulement\n🎯 J'attends une clôture M5 EN DESSOUS de {lo}"
    else:           plan = "😴 Tendance du jour neutre → pas de trade aujourd'hui"
    return (f"📏 RANGE D'ASIE — {pd.Timestamp(day).strftime('%d/%m')}\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"⬆️ Haut : {hi}\n⬇️ Bas  : {lo}\n📐 Taille : {round(hi - lo, 2)} $\n"
            f"━━━━━━━━━━━━━━━━━━\n{plan}\n🕐 Cassures acceptées jusqu'à {paris_label(ENTRY_END_MIN)} (Paris)")

def paris_label(minute_utc):
    from zoneinfo import ZoneInfo
    d = now_utc().replace(hour=minute_utc // 60, minute=minute_utc % 60, second=0, microsecond=0)
    return d.astimezone(ZoneInfo("Europe/Paris")).strftime("%Hh%M")

def format_entry(t):
    icon = "🟢" if t["dir"] == "BUY" else "🔴"
    return (f"🎯 ENTRE MAINTENANT — {icon} {t['dir']} XAUUSD\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"⚡ Prix : {t['entry']} (live)\n"
            f"🛑 SL   : {t['sl']}  ({t['risk']} $ — milieu du range d'Asie)\n"
            f"🎯 TP   : {t['tp']}  (2 R)\n"
            f"📦 Lot  : {lot_for(t['risk'])}  (risque ≈ {int(RISK_EUR)} €)\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"📊 Tendance + cassure du range d'Asie\n"
            f"⏰ Si ni SL ni TP : je te dis de fermer à {paris_label(CLOSE_MIN)} (Paris)\n"
            f"⚠️ Mets SL + TP tout de suite, puis ne touche plus à rien")

def format_result(t, why, r, price=None):
    gain = round(r * RISK_EUR)
    if why == "TP": return f"✅ TP TOUCHÉ — {t['dir']} {t['entry']} → {t['tp']}\n+2 R (≈ +{gain} €) 📒 note-le dans ton journal"
    if why == "SL": return f"❌ SL TOUCHÉ — {t['dir']} {t['entry']} → {t['sl']}\n−1 R (≈ {gain} €) 📒 note-le dans ton journal — c'est le jeu"
    sign = "+" if r >= 0 else ""
    return (f"⏰ FERME LE TRADE MAINTENANT — {t['dir']} {t['entry']} → ≈ {price}\n"
            f"{sign}{round(r, 2)} R (≈ {sign}{gain} €) 📒 note-le dans ton journal")

# ══════════════════════════════════════════════════════════════
#  Cycle
# ══════════════════════════════════════════════════════════════
async def send(bot, text):
    try:
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text)
    except Exception as e:
        log.error("Telegram: %s", e)

async def monitor_trade(bot):
    t = state["trade"]
    n = now_utc()
    df5 = closed(await get_candles("5min", 40))
    if df5 is not None:
        why, r = check_exit(t, df5)
        if why:
            await send(bot, format_result(t, why, r)); state["trade"] = None; return
    if minute_of_day(n) >= CLOSE_MIN or n.date() != t["opened"].date():
        live = await get_live_price()
        px = live if live is not None else (float(df5["close"].iloc[-1]) if df5 is not None else t["entry"])
        r = (px - t["entry"]) * t["s"] / t["risk"]
        await send(bot, format_result(t, "CLOSE", r, round(px, 2))); state["trade"] = None

async def cycle(bot):
    """Appelé 20 s après chaque clôture M5."""
    n = now_utc(); day = n.date(); mod = minute_of_day(n)
    if state["credit_day"] != day: state["credit_day"], state["credits"] = day, 0
    if state["trade"]:
        await monitor_trade(bot); return
    if n.weekday() >= 5: return
    if state["day"] != day:
        state.update(day=day, range=None, trend=None, done=False, last_bar=None)
    if mod < RANGE_END_MIN or state["done"]: return

    if state["range"] is None:
        df5 = closed(await get_candles("5min", 120))
        rng = asia_range(df5, day)
        if rng is None: return
        trend = daily_trend(closed(await get_candles("1day", 60)))
        state.update(range=rng, trend=trend)
        await send(bot, format_range(rng, trend, day))
        if trend == 0: state["done"] = True; return
        state["last_bar"] = pd.Timestamp(day) + pd.Timedelta(minutes=RANGE_END_MIN - 5)

    df5 = closed(await get_candles("5min", 30))
    if df5 is None: return
    new = df5[(df5["time"] > state["last_bar"]) & (df5["time"].dt.normalize() == pd.Timestamp(day))]
    for _, b in new.iterrows():
        bm = b["time"].hour * 60 + b["time"].minute
        state["last_bar"] = b["time"]
        if bm < RANGE_END_MIN or bm >= ENTRY_END_MIN: continue
        s = breakout(float(b["close"]), state["range"], state["trend"])
        if s:
            live = await get_live_price()
            entry = live if live is not None else float(b["close"])
            t = build_trade(s, entry, state["range"], b["time"])
            state["done"] = True
            if t is None:
                await send(bot, "⚠️ Cassure détectée mais le prix est déjà reparti de l'autre côté du SL — on laisse passer")
                return
            state["trade"] = t
            await send(bot, format_entry(t))
            log.info("ENTRÉE %s %s SL %s TP %s", t["dir"], t["entry"], t["sl"], t["tp"])
            return
    if mod >= ENTRY_END_MIN + 5 and not state["done"]:
        state["done"] = True
        await send(bot, "😴 Pas de cassure dans le sens de la tendance aujourd'hui — pas de trade, on attend demain")

async def main():
    bot = Bot(token=TELEGRAM_TOKEN, request=HTTPXRequest(read_timeout=30, connect_timeout=30, write_timeout=30))
    for _ in range(5):
        try:
            await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=(
                "🚀 XauBot Signal v26 TENDANCE + LIQUIDITÉ + CASSURE démarré\n\n"
                "1️⃣ Tendance du jour (EMA20 journalière) → sens du trade\n"
                f"2️⃣ Range d'Asie (jusqu'à {paris_label(RANGE_END_MIN)} Paris) → niveaux de liquidité\n"
                f"3️⃣ Cassure en clôture M5 jusqu'à {paris_label(ENTRY_END_MIN)} → 🎯 ENTRE MAINTENANT\n"
                f"🛑 SL milieu du range | 🎯 TP 2 R | ⏰ fermeture {paris_label(CLOSE_MIN)}\n"
                f"📦 Lot conseillé dans chaque signal (risque {int(RISK_EUR)} €)\n"
                "1 trade max par jour — signaux uniquement, aucun ordre automatique"))
            break
        except Exception as e:
            log.error("Startup: %s", e); await asyncio.sleep(10)
    log.info("XauBot v26 démarré")
    while True:
        await asyncio.sleep(seconds_to_next_m5())
        try:
            await cycle(bot)
        except Exception as e:
            log.exception("Erreur cycle: %s", e)
            await asyncio.sleep(30)

if __name__ == "__main__":
    asyncio.run(main())
