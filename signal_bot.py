# ══════════════════════════════════════════════════════════════
# XauBot — Signal Bot v27 CASSURE + RSI DANS LA TENDANCE
#
# 2 stratégies choisies par backtest sur 14 mois de bougies réelles Vantage (oct 2025 → nov 2026),
# vérifiées sur une période jamais vue (juil → nov 2026). Tendance du jour commune :
# clôture journalière d'hier au-dessus de son EMA20 → BUY seulement, en dessous → SELL seulement.
#
#  📈 A — CASSURE DU RANGE D'ASIE (v26)
#   range 00h00 → 07h00 UTC ; de 07h00 à 18h00 UTC, clôture M5 au-delà du range dans le sens
#   de la tendance → 🎯 | SL = milieu du range | TP = 2 R | 1 fois par jour
#
#  🔄 B — RSI DANS LA TENDANCE (style Larry Connors)
#   de 12h00 à 18h00 UTC, le RSI14 M5 ressort de la zone 30 (BUY) / 70 (SELL), dans le sens de
#   la tendance, prix dans la moitié 'discount' (BUY) / 'premium' (SELL) du range d'Asie → 🎯
#   SL = 1.5 ATR M15 | TP = 2 R | 1 h minimum entre deux signaux B
#
#  Commun : 1 seul trade ouvert à la fois, fermeture forcée à 21h00 UTC.
#  Signaux uniquement : le bot ne passe AUCUN ordre.
# ══════════════════════════════════════════════════════════════

import asyncio
import json
import logging
import os
import time
import httpx
import pandas as pd

from datetime import datetime, timezone
from datetime import datetime as _DT, date as _DATE
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
RSI_START_MIN    = 12 * 60     # stratégie B : 12h00 → 18h00 UTC
RSI_LOW, RSI_HIGH = 30, 70
RSI_SL_ATR       = 1.5
RSI_COOLDOWN_MIN = 60
MIN_RISK_USD     = 0.3
RISK_EUR         = float(os.environ.get("RISK_EUR", "100"))   # 1 % d'un compte de 10 000 €

STATE_FILE       = os.environ.get("STATE_FILE", "xaubot_state.json")   # sur Railway : volume monté sur /data → /data/state.json

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

state = {
    "day": None,        # jour UTC en cours
    "range": None,      # (haut, bas)
    "trend": None,      # 1 = BUY seulement, -1 = SELL seulement, 0 = rien
    "done": False,      # stratégie A : cassure du jour déjà vue
    "rsi_last": None,   # heure du dernier trade B
    "traded_today": 0,
    "last_bar": None,   # dernière bougie M5 analysée
    "trade": None,
    "credits": 0, "credit_day": None,
}

# ══════════════════════════════════════════════════════════════
#  Sauvegarde de l'état (un redémarrage ne fait pas oublier le trade en cours)
# ══════════════════════════════════════════════════════════════
def _enc(v):
    if isinstance(v, (pd.Timestamp, _DT)): return {"__t": v.isoformat()}
    if isinstance(v, dict): return {k: _enc(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [_enc(x) for x in v]
    if isinstance(v, _DATE): return {"__d": v.isoformat()}
    return v

def _dec(v):
    if isinstance(v, dict):
        if "__t" in v:
            t = pd.Timestamp(v["__t"]); return t.to_pydatetime() if t.tzinfo else t
        if "__d" in v: return pd.Timestamp(v["__d"]).date()
        return {k: _dec(x) for k, x in v.items()}
    if isinstance(v, list): return [_dec(x) for x in v]
    return v

def save_state():
    try:
        keep = {k: state[k] for k in ("day", "range", "trend", "done", "last_bar", "rsi_last", "traded_today", "trade")}
        with open(STATE_FILE, "w") as f: json.dump(_enc(keep), f)
    except Exception as e:
        log.error("save_state: %s", e)

def load_state():
    try:
        with open(STATE_FILE) as f: data = _dec(json.load(f))
        if data.get("range"): data["range"] = tuple(data["range"])
        state.update(data); return True
    except FileNotFoundError:
        return False
    except Exception as e:
        log.error("load_state: %s", e); return False

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
            "first_bar": pd.Timestamp(bar_time) + pd.Timedelta(minutes=5), "opened": now_utc(), "strat": "CASSURE"}

def check_exit(t, bars):
    """bars = bougies M5 fermées. On regarde celles d'après la bougie de cassure. SL prioritaire."""
    s = t["s"]
    for _, b in bars[bars["time"] >= t["first_bar"]].iterrows():
        if (s > 0 and b["low"] <= t["sl"]) or (s < 0 and b["high"] >= t["sl"]): return "SL", -1.0
        if (s > 0 and b["high"] >= t["tp"]) or (s < 0 and b["low"] <= t["tp"]): return "TP", TP_R
    return None, None

def rsi_series(close, p=14):
    d = close.diff(); g = d.clip(lower=0); l = -d.clip(upper=0)
    ag = g.ewm(alpha=1/p, adjust=False, min_periods=p).mean(); al = l.ewm(alpha=1/p, adjust=False, min_periods=p).mean()
    return 100 - 100 / (1 + ag / al.replace(0, float("nan")))

def rsi_signal(prev_rsi, cur_rsi):
    """1 si le RSI ressort au-dessus de 30, -1 s'il redescend sous 70, sinon 0."""
    if prev_rsi is None or cur_rsi is None or pd.isna(prev_rsi) or pd.isna(cur_rsi): return 0
    if cur_rsi > RSI_LOW and prev_rsi <= RSI_LOW: return 1
    if cur_rsi < RSI_HIGH and prev_rsi >= RSI_HIGH: return -1
    return 0

def rsi_filters_ok(s, trend, close, rng):
    """Tendance du jour dans le même sens + prix dans la bonne moitié du range d'Asie."""
    if trend * s <= 0 or rng is None: return False
    mid = (rng[0] + rng[1]) / 2
    return close < mid if s > 0 else close > mid

def atr_m15(df15, p=14):
    hi, lo, cl = df15["high"], df15["low"], df15["close"]
    tr = pd.concat([hi - lo, (hi - cl.shift()).abs(), (lo - cl.shift()).abs()], axis=1).max(axis=1)
    return float(tr.ewm(alpha=1/p, adjust=False).mean().iloc[-1])

def build_trade_rsi(s, entry, atr, bar_time):
    risk = RSI_SL_ATR * atr
    if risk <= MIN_RISK_USD: return None
    return {"dir": "BUY" if s > 0 else "SELL", "s": s, "entry": round(entry, 2), "sl": round(entry - s * risk, 2),
            "tp": round(entry + s * TP_R * risk, 2), "risk": round(risk, 2), "strat": "RSI",
            "first_bar": pd.Timestamp(bar_time) + pd.Timedelta(minutes=5), "opened": now_utc()}

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
    if t.get("strat") == "RSI":
        return (f"🎯 ENTRE MAINTENANT — {icon} {t['dir']} XAUUSD  🔄 RSI\n"
                f"━━━━━━━━━━━━━━━━━━\n"
                f"⚡ Prix : {t['entry']} (live)\n"
                f"🛑 SL   : {t['sl']}  ({t['risk']} $ — 1.5 ATR M15)\n"
                f"🎯 TP   : {t['tp']}  (2 R)\n"
                f"📦 Lot  : {lot_for(t['risk'])}  (risque ≈ {int(RISK_EUR)} €)\n"
                f"━━━━━━━━━━━━━━━━━━\n"
                f"📊 RSI ressort de {'30' if t['dir'] == 'BUY' else '70'} dans la tendance du jour\n"
                f"⏰ Si ni SL ni TP : je te dis de fermer à {paris_label(CLOSE_MIN)} (Paris)\n"
                f"⚠️ Mets SL + TP tout de suite, puis ne touche plus à rien")
    return (f"🎯 ENTRE MAINTENANT — {icon} {t['dir']} XAUUSD  📈 CASSURE\n"
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

async def open_trade(bot, t):
    state["trade"] = t; state["traded_today"] += 1
    await send(bot, format_entry(t))
    log.info("ENTRÉE %s %s %s SL %s TP %s", t.get("strat"), t["dir"], t["entry"], t["sl"], t["tp"])

async def cycle(bot):
    """Appelé 20 s après chaque clôture M5."""
    n = now_utc(); day = n.date(); mod = minute_of_day(n)
    if state["credit_day"] != day: state["credit_day"], state["credits"] = day, 0
    if state["trade"]:
        await monitor_trade(bot)
    if n.weekday() >= 5: return
    if state["day"] != day:
        state.update(day=day, range=None, trend=None, done=False, last_bar=None, rsi_last=None, traded_today=0)
    if mod < RANGE_END_MIN: return

    if state["range"] is None:
        df5 = closed(await get_candles("5min", 120))
        rng = asia_range(df5, day)
        if rng is None: return
        trend = daily_trend(closed(await get_candles("1day", 60)))
        state.update(range=rng, trend=trend)
        await send(bot, format_range(rng, trend, day))
        state["last_bar"] = pd.Timestamp(day) + pd.Timedelta(minutes=RANGE_END_MIN - 5)
    if state["trend"] == 0 or mod > ENTRY_END_MIN + 10: 
        if mod >= ENTRY_END_MIN + 5 and state["traded_today"] == 0 and state["last_bar"] is not None and state["trend"] != 0:
            await send(bot, "😴 Aucun signal aujourd'hui (ni cassure, ni RSI dans la tendance) — on attend demain")
            state["traded_today"] = -1
        return

    df5 = closed(await get_candles("5min", 250))
    if df5 is None: return
    rsi = rsi_series(df5["close"])
    today = df5["time"].dt.normalize() == pd.Timestamp(day)
    new_idx = df5.index[(df5["time"] > state["last_bar"]) & today]
    for k in new_idx:
        b = df5.loc[k]; bm = b["time"].hour * 60 + b["time"].minute
        state["last_bar"] = b["time"]
        if bm < RANGE_END_MIN or bm >= ENTRY_END_MIN: continue
        close = float(b["close"]); is_last = k == df5.index[-1]
        # A — cassure du range d'Asie (1 fois par jour, même si un trade est déjà ouvert)
        if not state["done"]:
            s = breakout(close, state["range"], state["trend"])
            if s:
                state["done"] = True
                if state["trade"] is None and is_last:
                    live = await get_live_price()
                    t = build_trade(s, live if live is not None else close, state["range"], b["time"])
                    if t: await open_trade(bot, t); continue
        # B — RSI dans la tendance
        if bm >= RSI_START_MIN and state["trade"] is None and is_last and k > df5.index[0]:
            s = rsi_signal(rsi.loc[k - 1], rsi.loc[k])
            if s and rsi_filters_ok(s, state["trend"], close, state["range"]):
                if state["rsi_last"] is None or (b["time"] - state["rsi_last"]) >= pd.Timedelta(minutes=RSI_COOLDOWN_MIN):
                    df15 = closed(await get_candles("15min", 200))
                    if df15 is None: continue
                    live = await get_live_price()
                    t = build_trade_rsi(s, live if live is not None else close, atr_m15(df15), b["time"])
                    if t:
                        state["rsi_last"] = b["time"]; await open_trade(bot, t)

async def main():
    restored = load_state()
    bot = Bot(token=TELEGRAM_TOKEN, request=HTTPXRequest(read_timeout=30, connect_timeout=30, write_timeout=30))
    for _ in range(5):
        try:
            await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=(
                "🚀 XauBot Signal v27 CASSURE + RSI DANS LA TENDANCE démarré\n\n"
                "1️⃣ Tendance du jour (EMA20 journalière) → sens du trade\n"
                f"2️⃣ Range d'Asie (jusqu'à {paris_label(RANGE_END_MIN)} Paris) → niveaux de liquidité\n"
                f"📈 A : cassure en clôture M5 jusqu'à {paris_label(ENTRY_END_MIN)} — SL milieu du range\n"
                f"🔄 B : RSI qui ressort de 30/70 de {paris_label(RSI_START_MIN)} à {paris_label(ENTRY_END_MIN)} — SL 1.5 ATR\n"
                f"🎯 TP 2 R | ⏰ fermeture {paris_label(CLOSE_MIN)} | 1 trade ouvert à la fois\n"
                f"📦 Lot conseillé dans chaque signal (risque {int(RISK_EUR)} €)\n"
                "Signaux uniquement — aucun ordre automatique"))
            break
        except Exception as e:
            log.error("Startup: %s", e); await asyncio.sleep(10)
    log.info("XauBot v27 démarré")
    if restored and state.get("trade"):
        t = state["trade"]
        await send(bot, f"♻️ Bot redémarré — je continue de suivre ton trade {t['dir']} {t['entry']} (SL {t['sl']} / TP {t['tp']})")
    while True:
        await asyncio.sleep(seconds_to_next_m5())
        try:
            await cycle(bot)
            save_state()
        except Exception as e:
            log.exception("Erreur cycle: %s", e)
            await asyncio.sleep(30)

if __name__ == "__main__":
    asyncio.run(main())
