# ══════════════════════════════════════════════════════════════
# XauBot — Signal Bot v25 REBOND + TENDANCE
#
# 2 stratégies, 1 seul trade à la fois (choisies par backtest, bougies réelles Vantage oct-nov 2026) :
#
#  ☀️ MATIN 7h-12h UTC (8h-13h Paris) — REBOND (≈ 87 % de trades gagnants au backtest)
#   - Tendance M15 : EMA20 > EMA50 → BUY seulement, EMA20 < EMA50 → SELL seulement
#   - Bougie M5 qui touche l'EMA20 M5 et la REJETTE : mèche ≥ 60 % de la bougie,
#     clôture du bon côté de l'EMA → « 🎯 ENTRE MAINTENANT »
#   - TP = 1.5 ATR M5 | SL = 3 ATR M5 (TP ≈ 0.5 R : beaucoup de petits gains)
#
#  🌙 APRÈS-MIDI / SOIR 12h-20h UTC (13h-21h Paris) — TENDANCE NY (v24, inchangée) :
#   - Tendance : M30 (EMA 20/50) + M15 (EMA 20/50, ADX ≥ 20, +DI/-DI, RSI 28-72) alignés
#   - Pullback : les 5 dernières bougies M5 ont clôturé du bon côté de l'EMA20 M5,
#     puis le prix revient TOUCHER l'EMA20 M5 → « 🎯 ENTRE MAINTENANT » (prix live)
#   - SL = 0.8 ATR M15 | TP unique = 1.5 R | on ne touche plus au SL
#   - Session New York uniquement : 12h-20h UTC (13h-21h Paris en hiver)
#   - Max 3 trades par jour, 1 seul trade à la fois
#   - Le bot suit le trade et annonce ✅ TP / ❌ SL (pour le journal)
#   - Signaux uniquement : le bot ne passe AUCUN ordre
# ══════════════════════════════════════════════════════════════

import asyncio
import logging
import os
import time
import httpx
import pandas as pd

from datetime import datetime, timezone, time as dtime
from zoneinfo import ZoneInfo
from telegram import Bot
from telegram.request import HTTPXRequest

TELEGRAM_TOKEN   = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
TWELVE_API_KEY   = os.environ["TWELVE_API_KEY"]

SYMBOL           = "XAU/USD"
REQUEST_TIMEOUT  = 15
SCAN_OFFSET_SEC  = 20      # scan 20 s après chaque clôture M5

# ── Stratégie (valeurs du backtest) ─────────────────────────────
ADX_MIN          = 20
RSI_MIN, RSI_MAX = 28, 72
EMA_M5           = 20
AWAY_BARS        = 5       # bougies M5 du bon côté de l'EMA avant le retour
SL_ATR           = 0.80    # SL = 0.8 ATR M15
TP_R             = 1.5     # TP unique à 1.5 R
MAX_TRADES_DAY   = int(os.environ.get("MAX_TRADES_DAY", "3"))
SESSION_START_UTC = int(os.environ.get("SESSION_START_UTC", "12"))
SESSION_END_UTC  = 20
MAX_HOLD_SEC     = 24 * 3600
MISSED_TOUCH_ATR = 0.30    # touche ratée entre 2 vérifs : entrée seulement si le prix est encore proche
RISK_EUR         = float(os.environ.get("RISK_EUR", "150"))

# ── REBOND du matin ─────────────────────────────────────────────
REB_START_UTC    = int(os.environ.get("REB_START_UTC", "7"))
REB_END_UTC      = 12
REB_WICK         = 0.60    # mèche ≥ 60 % de la bougie
REB_TP_ATR       = 1.5     # TP = 1.5 ATR M5
REB_SL_ATR       = 3.0     # SL = 3 ATR M5
REB_MAX_DAY      = 3

# ── Crédits Twelve Data (offre gratuite : 800/jour, 8/min) ──────
POLL_SEC         = 30      # prix live toutes les 30 s pendant l'attente
POLL_NEAR_ATR    = 1.0     # on ne surveille en direct que si le prix est à < 1 ATR de l'EMA
CREDIT_BUDGET    = 760     # au-delà, plus de surveillance live (scan M5 seulement)
HTF_CACHE_SEC    = 900

NY_TZ = ZoneInfo("America/New_York")
NEWS_BLACKOUTS = [
    (dtime(8, 25), dtime(8, 45)),   # stats US 8h30 ET (NFP, CPI…)
    (dtime(9, 0),  dtime(9, 45)),   # ouverture New York
]
FOMC_DATES = set(filter(None, os.environ.get("FOMC_DATES", "").split(",")))
FOMC_BLACKOUT = (dtime(13, 55), dtime(14, 45))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

state = {
    "trade": None,                 # trade en cours suivi par le bot
    "day": None, "count": 0, "count_reb": 0,   # trades du jour (UTC) : tendance / rebond
    "credit_day": None, "credits": 0,
    "armed": None,                 # contexte du cycle précédent (touche ratée)
}
_cache = {"m30": (None, 0), "m15": (None, None)}

# ══════════════════════════════════════════════════════════════
#  Horloge / session
# ══════════════════════════════════════════════════════════════
def now_utc():
    return datetime.now(timezone.utc)

def in_session():
    n = now_utc()
    return n.weekday() < 5 and SESSION_START_UTC <= n.hour < SESSION_END_UTC

def in_news_blackout():
    now_ny = datetime.now(NY_TZ); t = now_ny.time()
    if any(s <= t < e for s, e in NEWS_BLACKOUTS): return True
    return now_ny.strftime("%Y-%m-%d") in FOMC_DATES and FOMC_BLACKOUT[0] <= t < FOMC_BLACKOUT[1]

def seconds_to_next_m5():
    now = time.time()
    return max(5, (int(now) // 300 + 1) * 300 + SCAN_OFFSET_SEC - now)

def reset_daily():
    d = now_utc().date()
    if state["day"] != d: state["day"], state["count"], state["count_reb"] = d, 0, 0
    if state["credit_day"] != d: state["credit_day"], state["credits"] = d, 0

# ══════════════════════════════════════════════════════════════
#  Données Twelve Data
# ══════════════════════════════════════════════════════════════
async def get_candles(interval="5min", outputsize=200):
    state["credits"] += 1
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            r = await client.get("https://api.twelvedata.com/time_series",
                                 params={"symbol": SYMBOL, "interval": interval, "outputsize": outputsize,
                                         "apikey": TWELVE_API_KEY, "format": "JSON"})
            data = r.json()
        if "values" not in data:
            log.error("Twelve Data %s: %s", interval, data.get("message", data)); return None
        df = pd.DataFrame(data["values"]).rename(columns={"datetime": "time"})
        for c in ("open", "high", "low", "close"): df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.iloc[::-1].reset_index(drop=True).dropna(subset=["open", "high", "low", "close"])
        df["time"] = pd.to_datetime(df["time"])
        return df if len(df) >= 20 else None
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
    return None if df is None or len(df) < 5 else df.iloc[:-1].copy()   # dernière bougie = en formation

# ══════════════════════════════════════════════════════════════
#  Indicateurs (Wilder)
# ══════════════════════════════════════════════════════════════
def ema(s, p): return s.ewm(span=p, adjust=False).mean()
def wilder(s, p): return s.ewm(alpha=1/p, adjust=False, min_periods=p).mean()

def atr(df, p=14):
    hi, lo, cl = df["high"], df["low"], df["close"]
    tr = pd.concat([hi - lo, (hi - cl.shift()).abs(), (lo - cl.shift()).abs()], axis=1).max(axis=1)
    return wilder(tr, p)

def adx(df, p=14):
    up, down = df["high"].diff(), -df["low"].diff()
    pdm = pd.Series(0.0, index=df.index); mdm = pd.Series(0.0, index=df.index)
    pdm[(up > down) & (up > 0)] = up; mdm[(down > up) & (down > 0)] = down
    a = atr(df, p)
    pdi = 100 * wilder(pdm, p) / a.replace(0, pd.NA); mdi = 100 * wilder(mdm, p) / a.replace(0, pd.NA)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, pd.NA)
    return wilder(dx, p), pdi, mdi

def rsi(s, p=14):
    d = s.diff(); g = d.clip(lower=0); l = -d.clip(upper=0)
    rs = wilder(g, p) / wilder(l, p).replace(0, pd.NA)
    return (100 - 100 / (1 + rs)).fillna(100)

# ══════════════════════════════════════════════════════════════
#  Logique de la stratégie (fonctions pures → testables)
# ══════════════════════════════════════════════════════════════
def htf_state(df30):
    """M30 fermé : 2 = haussier fort, 1 = haussier, -1 = baissier, -2 = baissier fort, 0 = neutre."""
    if df30 is None or len(df30) < 210: return None
    c = df30["close"]; e20, e50, e200 = ema(c, 20).iloc[-1], ema(c, 50).iloc[-1], ema(c, 200).iloc[-1]
    p = c.iloc[-1]
    if p > e20 > e50 > e200: return 2
    if p < e20 < e50 < e200: return -2
    return 1 if e20 > e50 else (-1 if e20 < e50 else 0)

def m15_state(df15):
    if df15 is None or len(df15) < 60: return None
    c = df15["close"]; a, p, m = adx(df15, 14)
    return {"ef": float(ema(c, 20).iloc[-1]), "es": float(ema(c, 50).iloc[-1]), "atr": float(atr(df15, 14).iloc[-1]),
            "adx": float(a.iloc[-1]), "pdi": float(p.iloc[-1]), "mdi": float(m.iloc[-1]), "rsi": float(rsi(c, 14).iloc[-1])}

def trend_direction(htf, m15):
    """-1 = on cherche des SELL, 1 = des BUY, 0 = rien."""
    if htf is None or m15 is None or pd.isna(m15["adx"]): return 0
    if not (RSI_MIN <= m15["rsi"] <= RSI_MAX) or m15["adx"] < ADX_MIN: return 0
    if htf <= -1 and m15["ef"] < m15["es"] and m15["mdi"] > m15["pdi"]: return -1
    if htf >= 1 and m15["ef"] > m15["es"] and m15["pdi"] > m15["mdi"]: return 1
    return 0

def m5_context(df5, s):
    """df5 = bougies M5 FERMÉES. Retourne l'EMA à toucher pour la bougie suivante et si le pullback est valide."""
    c = df5["close"]; e = ema(c, EMA_M5)
    e_prev_bar = e.shift(1)                    # EMA connue à l'ouverture de chaque bougie
    away_ok = s != 0 and all((c.iloc[-k] - e_prev_bar.iloc[-k]) * s > 0 for k in range(1, AWAY_BARS + 1))
    return {"ema_next": float(e.iloc[-1]), "away_ok": away_ok, "last_close": float(c.iloc[-1]),
            "last_high": float(df5["high"].iloc[-1]), "last_low": float(df5["low"].iloc[-1]),
            "last_ema": float(e_prev_bar.iloc[-1]), "last_time": df5["time"].iloc[-1]}

def touch_entry(s, ema_level, live):
    """Le prix live touche l'EMA dans le sens du pullback → prix d'entrée, sinon None."""
    if live is None: return None
    return live if ((s < 0 and live >= ema_level) or (s > 0 and live <= ema_level)) else None

def build_trade(s, entry, atr15, extra=None, sl_dist=None, tp_dist=None):
    """Tendance : SL 0.8 ATR M15, TP 1.5 R. Rebond : distances données (3 ATR M5 / 1.5 ATR M5)."""
    risk = sl_dist if sl_dist is not None else SL_ATR * atr15
    tpd = tp_dist if tp_dist is not None else TP_R * risk
    t = {"dir": "SELL" if s < 0 else "BUY", "s": s, "entry": round(entry, 2),
         "sl": round(entry - s * risk, 2), "tp": round(entry + s * tpd, 2),
         "risk": round(risk, 2), "tp_r": round(tpd / risk, 2), "opened": now_utc(), "strat": "TENDANCE"}
    t.update(extra or {})
    return t

def rebond_signal(df5, trend15):
    """df5 = bougies M5 FERMÉES. Mèche ≥ 60 % qui rejette l'EMA20 M5 dans le sens de la tendance M15.
    Retourne (s, atr5) : s = 1 BUY, -1 SELL, 0 rien."""
    if df5 is None or len(df5) < 40 or not trend15: return 0, None
    e = float(ema(df5["close"], EMA_M5).iloc[-1]); a5 = float(atr(df5, 14).iloc[-1])
    o, h, l, c = (float(df5[k].iloc[-1]) for k in ("open", "high", "low", "close"))
    rng = max(h - l, 1e-9)
    lw = (min(o, c) - l) / rng; uw = (h - max(o, c)) / rng
    if trend15 > 0 and l <= e and c > e and lw >= REB_WICK: return 1, a5
    if trend15 < 0 and h >= e and c < e and uw >= REB_WICK: return -1, a5
    return 0, a5

def check_exit(trade, bars):
    """bars = bougies M5 fermées depuis l'entrée (la 1re contient l'entrée).
    SL prioritaire si SL et TP dans la même bougie. Sur la bougie d'entrée on ne regarde que le SL
    (le prix d'avant l'entrée ne compte pas pour le TP) — même règle que le backtest."""
    s = trade["s"]; entry_bar = pd.Timestamp(trade["opened"]).tz_localize(None).floor("5min")
    for _, b in bars.iterrows():
        if (s < 0 and b["high"] >= trade["sl"]) or (s > 0 and b["low"] <= trade["sl"]): return "SL"
        if b["time"] <= entry_bar and not trade.get("tp_first_bar"): continue
        if (s < 0 and b["low"] <= trade["tp"]) or (s > 0 and b["high"] >= trade["tp"]): return "TP"
    return None

def lot_for(risk_usd):
    if risk_usd <= 0: return 0.01
    return max(0.01, int(RISK_EUR / (risk_usd * 100) * 100) / 100)

# ══════════════════════════════════════════════════════════════
#  Messages
# ══════════════════════════════════════════════════════════════
HTF_TXT = {2: "HAUSSIÈRE FORTE", 1: "HAUSSIÈRE", -1: "BAISSIÈRE", -2: "BAISSIÈRE FORTE", 0: "NEUTRE"}

def format_entry(t):
    icon = "🔴" if t["dir"] == "SELL" else "🟢"
    if t.get("strat") == "REBOND":
        return (f"🎯 ENTRE MAINTENANT — {icon} {t['dir']} XAUUSD  ☀️ REBOND\n"
                f"━━━━━━━━━━━━━━━━━━\n"
                f"🕐 {t['opened'].strftime('%H:%M')} UTC\n"
                f"⚡ Prix : {t['entry']} (live)\n"
                f"🛑 SL   : {t['sl']}  ({t['risk']} $)\n"
                f"🎯 TP   : {t['tp']}  ({t['tp_r']} R) — petit TP, forte réussite\n"
                f"📦 Lot  : {lot_for(t['risk'])}  (risque ≈ {int(RISK_EUR)} €)\n"
                f"━━━━━━━━━━━━━━━━━━\n"
                f"🕯 Rejet de l'EMA20 M5 (mèche ≥ 60 %) dans la tendance M15\n"
                f"   trade {state['count_reb']}/{REB_MAX_DAY} rebond du jour\n"
                f"⚠️ Mets SL + TP tout de suite, puis ne touche plus à rien")
    return (f"🎯 ENTRE MAINTENANT — {icon} {t['dir']} XAUUSD  🌙 TENDANCE\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"🕐 {t['opened'].strftime('%H:%M')} UTC\n"
            f"⚡ Prix : {t['entry']} (live)\n"
            f"🛑 SL   : {t['sl']}  ({t['risk']} $)\n"
            f"🎯 TP   : {t['tp']}  (1.5 R) — TP unique\n"
            f"📦 Lot  : {lot_for(t['risk'])}  (risque ≈ {int(RISK_EUR)} €)\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"📊 Tendance M30 {HTF_TXT.get(t.get('htf'), '?')} | ADX M15 {t.get('adx', 0):.0f}\n"
            f"🕯 Retour sur l'EMA20 M5 | trade {state['count']}/{MAX_TRADES_DAY} du jour\n"
            f"⚠️ Mets SL + TP tout de suite, puis ne touche plus à rien")

def format_result(t, why):
    tp_r = t.get("tp_r", TP_R)
    gain = round(tp_r * RISK_EUR) if why == "TP" else -round(RISK_EUR)
    if why == "TP":
        return f"✅ TP TOUCHÉ — {t['dir']} {t['entry']} → {t['tp']}\n+{tp_r} R (≈ +{gain} €) 📒 note-le dans ton journal"
    if why == "SL":
        return f"❌ SL TOUCHÉ — {t['dir']} {t['entry']} → {t['sl']}\n−1 R (≈ {gain} €) 📒 note-le dans ton journal — c'est le jeu, on continue"
    return f"⌛ {t['dir']} {t['entry']} toujours ouvert — {why}"

# ══════════════════════════════════════════════════════════════
#  Cycle principal
# ══════════════════════════════════════════════════════════════
async def send(bot, text):
    try:
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text)
    except Exception as e:
        log.error("Telegram: %s", e)

async def get_context():
    """Tendance M30 (cache 15 min) + M15 (1 fois par bougie M15) + M5 fermées."""
    htf, ts = _cache["m30"]
    if htf is None or time.time() - ts > HTF_CACHE_SEC:
        htf = htf_state(closed(await get_candles("30min", 500)))
        _cache["m30"] = (htf, time.time())
    slot = int(time.time() // 900)
    m15, s15 = _cache["m15"]
    if m15 is None or s15 != slot:
        m15 = m15_state(closed(await get_candles("15min", 200)))
        _cache["m15"] = (m15, slot)
    df5 = closed(await get_candles("5min", 80))
    if df5 is None or m15 is None or htf is None: return None
    s = trend_direction(htf, m15)
    ctx = m5_context(df5, s); ctx.update(s=s, htf=htf, atr=m15["atr"], adx=m15["adx"])
    return ctx

async def open_trade(bot, s, entry, ctx):
    state["count"] += 1
    t = build_trade(s, entry, ctx["atr"], {"htf": ctx["htf"], "adx": ctx["adx"]})
    state["trade"] = t; state["armed"] = None
    await send(bot, format_entry(t))
    log.info("ENTRÉE %s %s SL %s TP %s", t["dir"], t["entry"], t["sl"], t["tp"])

async def monitor_trade(bot):
    t = state["trade"]
    df5 = closed(await get_candles("5min", 40))
    if df5 is None: return
    bars = df5[df5["time"] >= pd.Timestamp(t["opened"]).tz_localize(None).floor("5min")]
    why = check_exit(t, bars)
    if why is None:
        age = (now_utc() - t["opened"]).total_seconds()
        n = now_utc()
        if age > MAX_HOLD_SEC: why = "ferme-le au prix du marché (24 h)"
        elif n.weekday() == 4 and n.hour >= 20 and n.minute >= 30: why = "ferme-le avant le week-end"
    if why:
        await send(bot, format_result(t, why))
        state["trade"] = None

async def rebond_cycle(bot):
    """Matin : cherche une bougie de rejet de l'EMA20 M5 dans la tendance M15."""
    if state["count_reb"] >= REB_MAX_DAY: return
    slot = int(time.time() // 900)
    m15, s15 = _cache["m15"]
    if m15 is None or s15 != slot:
        m15 = m15_state(closed(await get_candles("15min", 200)))
        _cache["m15"] = (m15, slot)
    if m15 is None: return
    trend = 1 if m15["ef"] > m15["es"] else (-1 if m15["ef"] < m15["es"] else 0)
    s, a5 = rebond_signal(closed(await get_candles("5min", 80)), trend)
    if not s: return
    live = await get_live_price()
    if live is None: return
    state["count_reb"] += 1
    t = build_trade(s, live, None, {"strat": "REBOND", "tp_first_bar": True},
                    sl_dist=REB_SL_ATR * a5, tp_dist=REB_TP_ATR * a5)
    state["trade"] = t
    await send(bot, format_entry(t))
    log.info("REBOND %s %s SL %s TP %s", t["dir"], t["entry"], t["sl"], t["tp"])

async def cycle(bot):
    """Appelé 20 s après chaque clôture M5."""
    reset_daily()
    if state["trade"]:
        await monitor_trade(bot)
        if state["trade"]: return
    n = now_utc(); bar_start = n - pd.Timedelta(minutes=5, seconds=SCAN_OFFSET_SEC)
    if n.weekday() < 5 and REB_START_UTC <= bar_start.hour < REB_END_UTC:
        await rebond_cycle(bot); return
    if not in_session() or in_news_blackout() or state["count"] >= MAX_TRADES_DAY:
        state["armed"] = None; return

    ctx = await get_context()
    if ctx is None: return

    # Touche ratée entre deux vérifications pendant la bougie qui vient de fermer
    prev = state["armed"]
    if prev and prev["s"] == ctx["s"]:
        touched = (prev["s"] < 0 and ctx["last_high"] >= prev["ema_next"]) or (prev["s"] > 0 and ctx["last_low"] <= prev["ema_next"])
        if touched:
            live = await get_live_price()
            if live is not None and abs(live - prev["ema_next"]) <= MISSED_TOUCH_ATR * ctx["atr"]:
                await open_trade(bot, ctx["s"], live, ctx); return
    state["armed"] = None

    if not ctx["away_ok"]: return
    state["armed"] = {"s": ctx["s"], "ema_next": ctx["ema_next"]}

    near = abs(ctx["last_close"] - ctx["ema_next"]) <= POLL_NEAR_ATR * ctx["atr"]
    if not near or state["credits"] >= CREDIT_BUDGET: return
    while seconds_to_next_m5() > POLL_SEC + 5:
        await asyncio.sleep(POLL_SEC)
        live = await get_live_price()
        entry = touch_entry(ctx["s"], ctx["ema_next"], live)
        if entry is not None:
            await open_trade(bot, ctx["s"], entry, ctx); return

async def main():
    bot = Bot(token=TELEGRAM_TOKEN, request=HTTPXRequest(read_timeout=30, connect_timeout=30, write_timeout=30))
    for _ in range(5):
        try:
            await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=(
                "🚀 XauBot Signal v25 REBOND + TENDANCE démarré\n\n"
                f"☀️ {REB_START_UTC}h-{REB_END_UTC}h UTC (8h-13h Paris) : REBOND sur l'EMA20 M5\n"
                f"   TP {REB_TP_ATR} ATR M5 | SL {REB_SL_ATR} ATR M5 | max {REB_MAX_DAY}/jour\n"
                f"🌙 {SESSION_START_UTC}h-{SESSION_END_UTC}h UTC (13h-21h Paris) : TENDANCE NY\n"
                f"   SL {SL_ATR} ATR M15 | TP unique {TP_R} R | max {MAX_TRADES_DAY}/jour\n"
                "🔒 1 seul trade à la fois\n"
                f"📦 Lot conseillé dans chaque signal (risque {int(RISK_EUR)} €)\n"
                "✅/❌ Le bot t'annonce le résultat de chaque trade\n"
                "⚠️ Signaux uniquement — aucun ordre automatique"))
            break
        except Exception as e:
            log.error("Startup: %s", e); await asyncio.sleep(10)
    log.info("XauBot v25 démarré")
    while True:
        await asyncio.sleep(seconds_to_next_m5())
        try:
            await cycle(bot)
        except Exception as e:
            log.exception("Erreur cycle: %s", e)
            await asyncio.sleep(30)

if __name__ == "__main__":
    asyncio.run(main())
