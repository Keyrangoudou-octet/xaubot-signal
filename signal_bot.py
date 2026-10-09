# XauBot — Signal Bot v23.2 PRO
#
# Fixes v23.2 :
#   - get_structure() : BOS comparé aux bougies PRÉCÉDENTES (avant : impossible à déclencher)
#   - Plantage SELL limit corrigé (DataFrame évalué en booléen)
#   - M5 score minimum réellement à 50 (constante M5_MIN_SCORE)
#   - Rejet SELL : plus d'annulation au moment du signal, ordre envoyé "⏳ en attente"
#   - Fibonacci calculé selon la direction (SELL = retracement depuis le bas)
#   - R:R calculé sur une vraie cible (swing high / swing low)
#   - Filtre news / ouverture NY en heure de New York (gère l'heure d'été)
#   - EMA 200 M30 : 500 bougies chargées
#
# Conservé de v23.1 : httpx async, closed_df (anti-repainting), Wilder RSI/ADX/ATR,
#   filtre RSI extrême (<28 / >72), horaire 7h-20h UTC.

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

SCAN_INTERVAL    = 180
SIGNAL_COOLDOWN  = 1800
FUTURES_OFFSET   = 0
ENTRY_BUFFER     = 1.50
MIN_CONFIDENCE   = 70      # tester 65 si toujours trop peu de signaux
M5_MIN_SCORE     = 50
REQUEST_TIMEOUT  = 15
REJECT_RANGE     = 5.0
REJECT_LOOKBACK  = 6

# ── Filtre news (heure de New York, heure d'été gérée automatiquement) ──
NY_TZ = ZoneInfo("America/New_York")
NEWS_BLACKOUTS = [
    (dtime(8, 25), dtime(8, 45)),   # stats US 8h30 ET (NFP, CPI, PPI, jobless claims...)
    (dtime(9, 0),  dtime(9, 45)),   # ouverture New York
]
# Jours FOMC : bloque 13h55-14h45 ET (décision 14h00 + conférence 14h30).
# Format "YYYY-MM-DD" — à compléter avec le calendrier officiel de la Fed.
FOMC_DATES = set(filter(None, os.environ.get("FOMC_DATES", "").split(",")))
FOMC_BLACKOUT = (dtime(13, 55), dtime(14, 45))

XAUUSD_CONFIG = {
    "symbol":        "XAU/USD",
    "label":         "XAUUSD",
    "ema_fast":      20,
    "ema_slow":      50,
    "ema_trend":     200,
    "adx_period":    14,
    "adx_min":       23,
    "rsi_period":    14,
    "atr_period":    14,
    "swing_lookback":20,
    "fibo_lookback": 60,
    "min_rr":        1.5,
    "fvg_min_atr":   0.15,
    "ob_impulse_atr":1.2,
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

def is_market_open():
    now     = datetime.now(timezone.utc)
    weekday = now.weekday()
    hour    = now.hour
    if weekday == 5: return False
    if weekday == 6: return False
    if weekday == 4 and hour >= 20: return False
    return 7 <= hour < 20

def in_news_blackout():
    now_ny = datetime.now(NY_TZ)
    t = now_ny.time()
    for start, end in NEWS_BLACKOUTS:
        if start <= t < end:
            return True
    if now_ny.strftime("%Y-%m-%d") in FOMC_DATES:
        if FOMC_BLACKOUT[0] <= t < FOMC_BLACKOUT[1]:
            return True
    return False

async def get_candles(symbol, interval="5min", outputsize=200):
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            r = await client.get(
                "https://api.twelvedata.com/time_series",
                params={"symbol": symbol, "interval": interval,
                        "outputsize": outputsize, "apikey": TWELVE_API_KEY, "format": "JSON"}
            )
            data = r.json()
        if "values" not in data:
            log.error("Twelve Data %s: %s", symbol, data.get("message", data))
            return None
        df = pd.DataFrame(data["values"]).rename(columns={"datetime": "time"})
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.iloc[::-1].reset_index(drop=True)
        df = df.dropna(subset=["open", "high", "low", "close"])
        return df if len(df) >= 20 else None
    except Exception as e:
        log.error("get_candles %s: %s", symbol, e)
        return None

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def wilder_smooth(series, period):
    return series.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

def atr(df, period=14):
    hi, lo, cl = df["high"], df["low"], df["close"]
    tr = pd.concat([hi-lo, (hi-cl.shift()).abs(), (lo-cl.shift()).abs()], axis=1).max(axis=1)
    return wilder_smooth(tr, period)

def adx(df, period=14):
    up, down = df["high"].diff(), -df["low"].diff()
    pdm = pd.Series(0.0, index=df.index)
    mdm = pd.Series(0.0, index=df.index)
    pdm[(up > down) & (up > 0)]   = up
    mdm[(down > up) & (down > 0)] = down
    atr_v = atr(df, period)
    pdi = 100 * wilder_smooth(pdm, period) / atr_v.replace(0, pd.NA)
    mdi = 100 * wilder_smooth(mdm, period) / atr_v.replace(0, pd.NA)
    dx  = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, pd.NA)
    return wilder_smooth(dx, period), pdi, mdi

def rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    rs    = wilder_smooth(gain, period) / wilder_smooth(loss, period).replace(0, pd.NA)
    return (100 - (100 / (1 + rs))).fillna(100)

def closed_df(df):
    if df is None or len(df) < 5: return df
    return df.iloc[:-1].copy()

def get_structure(df, lookback=20):
    # FIX v23.2 : on compare le close actuel aux bougies PRÉCÉDENTES uniquement.
    # Avant, la bougie actuelle était incluse → close > sa propre high impossible.
    if len(df) < lookback + 5: return None
    prev    = df.iloc[-(lookback+1):-1]
    current = float(df["close"].iloc[-1])
    if current > float(prev["high"].max()): return "BOS_BULL"
    if current < float(prev["low"].min()):  return "BOS_BEAR"
    return "RANGE"

def double_impulse(df, atr_value):
    if len(df) < 5: return False, False
    c1, c2 = df.iloc[-1], df.iloc[-2]
    b1 = abs(c1["close"] - c1["open"])
    b2 = abs(c2["close"] - c2["open"])
    bull = c1["close"] > c1["open"] and c2["close"] > c2["open"] and b1 > atr_value*0.3 and b2 > atr_value*0.3
    bear = c1["close"] < c1["open"] and c2["close"] < c2["open"] and b1 > atr_value*0.3 and b2 > atr_value*0.3
    return bull, bear

def detect_breakout(df, atr_value, lookback=20):
    if len(df) < lookback + 3: return None
    current  = df.iloc[-1]
    previous = df.iloc[-(lookback+1):-1]
    body     = abs(current["close"] - current["open"])
    if current["close"] > float(previous["high"].max()) and body > atr_value*0.8: return "BULL"
    if current["close"] < float(previous["low"].min())  and body > atr_value*0.8: return "BEAR"
    return None

def detect_pattern(df):
    if len(df) < 4: return None
    cur, prev = df.iloc[-1], df.iloc[-2]
    o, h, l, c = float(cur["open"]), float(cur["high"]), float(cur["low"]), float(cur["close"])
    po, pc = float(prev["open"]), float(prev["close"])
    total = h - l
    if total <= 0: return None
    body  = abs(c - o)
    upper = h - max(o, c)
    lower = min(o, c) - l
    if body/total < 0.10:                                         return "DOJI"
    if lower/total > 0.55 and body/total < 0.35 and c >= o:      return "PIN_BULL"
    if upper/total > 0.55 and body/total < 0.35 and c <= o:      return "PIN_BEAR"
    if c > o and pc < po and c >= po and o <= pc:                 return "ENGULFING_BULL"
    if c < o and pc > po and c <= po and o >= pc:                 return "ENGULFING_BEAR"
    return None

def detect_fvg(df, atr_value=None, lookback=30, min_atr_ratio=0.15):
    if len(df) < 5: return None
    start = max(2, len(df) - lookback)
    best  = None
    for i in range(start, len(df)):
        c1, c3 = df.iloc[i-2], df.iloc[i]
        gl, gh = float(c1["high"]), float(c3["low"])
        if gh > gl and (atr_value is None or gh-gl >= atr_value*min_atr_ratio):
            best = ("BULL", round(gl,2), round(gh,2), i)
        gl, gh = float(c3["high"]), float(c1["low"])
        if gh > gl and (atr_value is None or gh-gl >= atr_value*min_atr_ratio):
            best = ("BEAR", round(gl,2), round(gh,2), i)
    if best is None: return None
    direction, low, high, index = best
    future = df.iloc[index+1:]
    if len(future) > 0:
        if direction == "BULL" and future["low"].min() <= low:  return None
        if direction == "BEAR" and future["high"].max() >= high: return None
    return (direction, low, high)

def detect_ob(df, atr_value, lookback=40, impulse_mult=1.2):
    if len(df) < 10: return None
    candidates = []
    for i in range(max(2, len(df)-lookback), len(df)-3):
        c = df.iloc[i]
        o, cl, h, l = float(c["open"]), float(c["close"]), float(c["high"]), float(c["low"])
        future = df.iloc[i+1:i+4]
        if len(future) < 2: continue
        if cl < o and (float(future["high"].max()) - h) >= atr_value*impulse_mult:
            candidates.append(("BULL", round(l,2), round(h,2), i))
        if cl > o and (l - float(future["low"].min())) >= atr_value*impulse_mult:
            candidates.append(("BEAR", round(l,2), round(h,2), i))
    if not candidates: return None
    direction, low, high, index = candidates[-1]
    future = df.iloc[index+1:]
    if direction == "BULL" and future["low"].min() <= low:  return None
    if direction == "BEAR" and future["high"].max() >= high: return None
    return (direction, low, high)

def fibo_range(df, lookback=60):
    r = df.tail(lookback)
    return float(r["high"].max()), float(r["low"].min())

def fibonacci_levels(high, low, direction):
    # FIX v23.2 : BUY = retracement depuis le haut, SELL = retracement depuis le bas.
    d = high - low
    if d <= 0: return None
    r = {"38.2": .382, "50.0": .5, "61.8": .618}
    if direction == "BUY":
        return {k: round(high - d*v, 2) for k, v in r.items()}
    return {k: round(low + d*v, 2) for k, v in r.items()}

def check_zone_rejection(df5, entry, direction):
    n = len(df5)
    lookback = min(REJECT_LOOKBACK, n-1)
    for i in range(n-2, n-2-lookback, -1):
        if i < 0: break
        o = float(df5["open"].iloc[i]); c = float(df5["close"].iloc[i])
        h = float(df5["high"].iloc[i]); l = float(df5["low"].iloc[i])
        if direction == "SELL" and c < o and h >= entry - REJECT_RANGE: return True
        if direction == "BUY"  and c > o and l <= entry + REJECT_RANGE: return True
    return False

async def get_htf_analysis(symbol):
    # FIX v23.2 : 500 bougies pour une EMA 200 fiable (1 crédit API quel que soit outputsize)
    df = await get_candles(symbol, interval="30min", outputsize=500)
    if df is None: return None
    df = closed_df(df)
    if len(df) < 210: return None
    df["ema20"]  = ema(df["close"], 20)
    df["ema50"]  = ema(df["close"], 50)
    df["ema200"] = ema(df["close"], 200)
    p = float(df["close"].iloc[-1])
    e20, e50, e200 = float(df["ema20"].iloc[-1]), float(df["ema50"].iloc[-1]), float(df["ema200"].iloc[-1])
    if p > e20 > e50 > e200: return "BULL_STRONG"
    if p < e20 < e50 < e200: return "BEAR_STRONG"
    if e20 > e50: return "BULL"
    if e20 < e50: return "BEAR"
    return "NEUTRAL"

async def get_m5_confirmation(symbol, direction):
    df = await get_candles(symbol, interval="5min", outputsize=120)
    if df is None: return None
    df = closed_df(df)
    if len(df) < 60: return None
    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)
    atr_v       = float(atr(df).iloc[-1])
    adx_v, pdi, mdi = adx(df)
    pattern = detect_pattern(df)
    score   = 0
    if direction == "BUY":
        if df["ema20"].iloc[-1] > df["ema50"].iloc[-1]: score += 25
        if pattern in ("ENGULFING_BULL", "PIN_BULL"):    score += 20
        if pdi.iloc[-1] > mdi.iloc[-1]:                  score += 20
    else:
        if df["ema20"].iloc[-1] < df["ema50"].iloc[-1]: score += 25
        if pattern in ("ENGULFING_BEAR", "PIN_BEAR"):    score += 20
        if mdi.iloc[-1] > pdi.iloc[-1]:                  score += 20
    if adx_v.iloc[-1] >= 20: score += 15
    return {"price": round(float(df["close"].iloc[-1]),2), "score": score,
            "pattern": pattern, "atr": atr_v, "df5": df}

def calculate_confidence(direction, htf, ef, es, price,
                          pdi, mdi, adx_value, rsi_value,
                          pattern, fvg, ob, structure, breakout):
    score, reasons = 0, []
    if direction == "BUY":
        if htf == "BULL_STRONG": score += 20; reasons.append("M30 tendance forte")
        elif htf == "BULL":      score += 12; reasons.append("M30 haussier")
        if ef > es:              score += 15; reasons.append("EMA M15 bullish")
        if pdi > mdi:            score += 15; reasons.append("+DI dominant")
        if 50 <= rsi_value <= 68: score += 8; reasons.append("RSI bullish sain")
        if pattern in ("ENGULFING_BULL", "PIN_BULL"): score += 7
        if structure == "BOS_BULL": score += 10; reasons.append("Break structure bullish")
        if fvg and fvg[0] == "BULL": score += 7;  reasons.append("FVG aligné")
        if ob  and ob[0]  == "BULL": score += 10; reasons.append("OB aligné")
    else:
        if htf == "BEAR_STRONG": score += 20; reasons.append("M30 tendance forte")
        elif htf == "BEAR":      score += 12; reasons.append("M30 baissier")
        if ef < es:              score += 15; reasons.append("EMA M15 bearish")
        if mdi > pdi:            score += 15; reasons.append("-DI dominant")
        if 32 <= rsi_value <= 50: score += 8; reasons.append("RSI bearish sain")
        if pattern in ("ENGULFING_BEAR", "PIN_BEAR"): score += 7
        if structure == "BOS_BEAR": score += 10; reasons.append("Break structure bearish")
        if fvg and fvg[0] == "BEAR": score += 7;  reasons.append("FVG aligné")
        if ob  and ob[0]  == "BEAR": score += 10; reasons.append("OB aligné")
    if adx_value >= 30:   score += 15; reasons.append("ADX fort")
    elif adx_value >= 23: score += 8;  reasons.append("ADX valide")
    if breakout: score += 10; reasons.append("Breakout confirmé")
    return min(score, 100), reasons

async def analyze_xauusd():
    cfg = XAUUSD_CONFIG
    htf, df = await asyncio.gather(
        get_htf_analysis(cfg["symbol"]),
        get_candles(cfg["symbol"], interval="15min", outputsize=200),
    )
    if not htf or df is None: return None
    df = closed_df(df)
    if len(df) < 100: return None

    df["ef"] = ema(df["close"], cfg["ema_fast"])
    df["es"] = ema(df["close"], cfg["ema_slow"])
    atr_v    = float(atr(df, cfg["atr_period"]).iloc[-1])
    adx_s, pdi, mdi = adx(df, cfg["adx_period"])
    adx_v = float(adx_s.iloc[-1]); pdi_v = float(pdi.iloc[-1]); mdi_v = float(mdi.iloc[-1])
    rsi_v = float(rsi(df["close"], cfg["rsi_period"]).iloc[-1])
    price = float(df["close"].iloc[-1])
    ef    = float(df["ef"].iloc[-1]); es = float(df["es"].iloc[-1])

    structure        = get_structure(df, cfg["swing_lookback"])
    breakout         = detect_breakout(df, atr_v, cfg["swing_lookback"])
    bull_imp, bear_imp = double_impulse(df, atr_v)
    pattern          = detect_pattern(df)
    fvg              = detect_fvg(df, atr_v, min_atr_ratio=cfg["fvg_min_atr"])
    ob               = detect_ob(df, atr_v, impulse_mult=cfg["ob_impulse_atr"])
    swing_h, swing_l = fibo_range(df, cfg["fibo_lookback"])

    # ── Filtre RSI dur : extensions extrêmes ──────────────────
    if rsi_v < 28:
        log.info("Signal bloqué : RSI extrême bas %.1f (oversold)", rsi_v)
        return None
    if rsi_v > 72:
        log.info("Signal bloqué : RSI extrême haut %.1f (overbought)", rsi_v)
        return None

    buy_ok  = (htf in ("BULL","BULL_STRONG") and ef > es and pdi_v > mdi_v
               and adx_v >= cfg["adx_min"]
               and (bull_imp or breakout == "BULL" or structure == "BOS_BULL"))
    sell_ok = (htf in ("BEAR","BEAR_STRONG") and ef < es and mdi_v > pdi_v
               and adx_v >= cfg["adx_min"]
               and (bear_imp or breakout == "BEAR" or structure == "BOS_BEAR"))

    direction = "BUY" if buy_ok else ("SELL" if sell_ok else None)
    if direction is None:
        log.info("Pas de setup (htf=%s ef>es=%s +DI>-DI=%s adx=%.1f struct=%s brk=%s)",
                 htf, ef > es, pdi_v > mdi_v, adx_v, structure, breakout)
        return None

    # FIX v23.2 : Fibo calculé selon la direction
    fib = fibonacci_levels(swing_h, swing_l, direction)

    confidence, reasons = calculate_confidence(
        direction, htf, ef, es, price, pdi_v, mdi_v, adx_v, rsi_v,
        pattern, fvg, ob, structure, breakout)

    if confidence < MIN_CONFIDENCE:
        log.info("Signal %s rejeté : confidence %s < %s", direction, confidence, MIN_CONFIDENCE)
        return None

    m5 = await get_m5_confirmation(cfg["symbol"], direction)
    if m5 is None or m5["score"] < M5_MIN_SCORE:
        log.info("Signal %s rejeté : M5 faible (%s)", direction, m5["score"] if m5 else None)
        return None

    stop_distance = atr_v * 1.5
    sl   = price - stop_distance if direction == "BUY" else price + stop_distance
    risk = abs(price - sl)
    if direction == "BUY":
        tp1, tp2, tp3 = price+risk, price+risk*1.5, price+risk*2.5
    else:
        tp1, tp2, tp3 = price-risk, price-risk*1.5, price-risk*2.5

    return {
        "direction": direction, "price": round(price,2), "sl": round(sl,2),
        "tp1": round(tp1,2), "tp2": round(tp2,2), "tp3": round(tp3,2),
        "adx": round(adx_v,1), "pdi": round(pdi_v,1), "mdi": round(mdi_v,1),
        "rsi": round(rsi_v,1), "htf": htf, "pattern": pattern,
        "structure": structure, "breakout": breakout, "fvg": fvg, "ob": ob,
        "fib": fib, "swing_h": swing_h, "swing_l": swing_l,
        "atr": atr_v, "confidence": confidence, "reasons": reasons,
        "m5_score": m5["score"], "m5_pattern": m5["pattern"], "m5_df5": m5["df5"],
        "signal_type": "BREAKOUT_CONF" if breakout else "SIGNAL",
    }

def calc_limit_entry(signal):
    direction = signal["direction"]; atr_v = signal["atr"]
    fib = signal["fib"]; fvg = signal["fvg"]; ob = signal["ob"]; price = signal["price"]
    candidates = []
    if fib:
        for lvl_name in ["38.2","50.0","61.8"]:
            lvl = fib[lvl_name]
            if direction == "BUY"  and lvl >= price: continue
            if direction == "SELL" and lvl <= price: continue
            score = {"38.2":15,"50.0":20,"61.8":25}[lvl_name]; conf = []
            if fvg:
                if direction == "BUY"  and fvg[0]=="BULL" and fvg[1]-8<=lvl<=fvg[2]+8: score+=25; conf.append("FVG")
                if direction == "SELL" and fvg[0]=="BEAR" and fvg[1]-8<=lvl<=fvg[2]+8: score+=25; conf.append("FVG")
            if ob:
                if direction == "BUY"  and ob[0]=="BULL" and ob[1]-8<=lvl<=ob[2]+8: score+=30; conf.append("OB")
                if direction == "SELL" and ob[0]=="BEAR" and ob[1]-8<=lvl<=ob[2]+8: score+=30; conf.append("OB")
            sl = lvl-atr_v*0.8 if direction=="BUY" else lvl+atr_v*0.8
            candidates.append({"limit":round(lvl,2),"sl":round(sl,2),"source":"FIBO "+lvl_name,"score":score,"confluence":conf})
    if ob:
        if direction=="BUY"  and ob[0]=="BULL": candidates.append({"limit":ob[2],"sl":ob[1]-atr_v*0.4,"source":"ORDER BLOCK","score":65,"confluence":["OB"]})
        if direction=="SELL" and ob[0]=="BEAR": candidates.append({"limit":ob[1],"sl":ob[2]+atr_v*0.4,"source":"ORDER BLOCK","score":65,"confluence":["OB"]})
    if fvg:
        if direction=="BUY"  and fvg[0]=="BULL": candidates.append({"limit":fvg[2]+ENTRY_BUFFER,"sl":fvg[1]-atr_v*0.4,"source":"FVG","score":60,"confluence":["FVG"]})
        if direction=="SELL" and fvg[0]=="BEAR": candidates.append({"limit":fvg[1]-ENTRY_BUFFER,"sl":fvg[2]+atr_v*0.4,"source":"FVG","score":60,"confluence":["FVG"]})
    if not candidates: return None
    best  = sorted(candidates, key=lambda x: x["score"], reverse=True)[0]
    entry = best["limit"]; sl = best["sl"]; risk = abs(entry-sl)
    if risk <= 0: return None

    # FIX v23.2 : R:R mesuré sur une vraie cible structurelle (swing high / swing low)
    target = signal["swing_h"] if direction == "BUY" else signal["swing_l"]
    reward = (target - entry) if direction == "BUY" else (entry - target)
    rr = reward / risk
    if rr < XAUUSD_CONFIG["min_rr"]:
        log.info("Limit %s @ %s rejeté : R:R %.2f vers cible %.2f", direction, entry, rr, target)
        return None
    best["rr"] = round(rr,2); best["target"] = round(target,2)
    best["confidence"] = "HIGH" if best["score"]>=70 else "MEDIUM"
    return best

def format_message(signal):
    d = signal["direction"]
    icon = "🟢" if signal["confidence"] >= 85 else "🟡"
    now  = datetime.now(timezone.utc).strftime("%H:%M UTC")
    msg  = (
        f"{'🟢' if d=='BUY' else '🔴'} SIGNAL {d} — XAUUSD\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"🕐 {now}\n"
        f"📍 Entry : {signal['price']}\n"
        f"🛑 SL    : {signal['sl']}\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"🎯 TP1   : {signal['tp1']} (1R)\n"
        f"🎯 TP2   : {signal['tp2']} (1.5R)\n"
        f"🎯 TP3   : {signal['tp3']} (2.5R)\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"📊 ADX   : {signal['adx']}\n"
        f"📈 +DI   : {signal['pdi']}\n"
        f"📉 -DI   : {signal['mdi']}\n"
        f"📉 RSI   : {signal['rsi']}\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"🧭 M30   : {signal['htf']}\n"
        f"🏗 Structure : {signal['structure']}\n"
        f"🔬 M5 score  : {signal['m5_score']}/80\n"
    )
    if signal["pattern"]:    msg += f"🕯 Pattern M15 : {signal['pattern']}\n"
    if signal["m5_pattern"]: msg += f"🕯 Pattern M5  : {signal['m5_pattern']}\n"
    if signal["fvg"]:        msg += f"📐 FVG : {signal['fvg'][0]} [{signal['fvg'][1]} - {signal['fvg'][2]}]\n"
    if signal["ob"]:         msg += f"📦 OB  : {signal['ob'][0]} [{signal['ob'][1]} - {signal['ob'][2]}]\n"
    msg += f"━━━━━━━━━━━━━━━━━━\n{icon} CONFIANCE : {signal['confidence']}/100\n"
    if signal["reasons"]: msg += "✅ " + " | ".join(signal["reasons"])
    return msg

def format_limit_message(signal, limit, sell_rejected=False):
    if limit is None: return None
    d = signal["direction"]; entry = limit["limit"]; sl = limit["sl"]; risk = abs(entry-sl)
    tp1 = entry+risk if d=="BUY" else entry-risk
    tp2 = entry+risk*1.5 if d=="BUY" else entry-risk*1.5
    tp3 = entry+risk*2.5 if d=="BUY" else entry-risk*2.5
    msg = (
        f"📌 ORDRE LIMIT — XAUUSD\n━━━━━━━━━━━━━━━━━━\n"
        f"{'🟢' if d=='BUY' else '🔴'} {d} LIMIT\n\n"
        f"📍 ENTRY : {entry}\n🛑 SL    : {round(sl,2)}\n━━━━━━━━━━━━━━━━━━\n"
        f"🎯 TP1   : {round(tp1,2)}\n🎯 TP2   : {round(tp2,2)}\n🎯 TP3   : {round(tp3,2)}\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"📐 Source : {limit['source']}\n"
        f"🔥 Confluence : {', '.join(limit['confluence']) or 'FIBO'}\n"
        f"📊 Score zone : {limit['score']}/100\n"
        f"⚖️ RR vers {'swing high' if d=='BUY' else 'swing low'} ({limit['target']}) : {limit['rr']}\n"
        f"━━━━━━━━━━━━━━━━━━\n"
    )
    if d == "SELL":
        msg += "🕯 Rejet M5 : ✅ confirmé\n" if sell_rejected else "🕯 Rejet M5 : ⏳ en attente — attends une bougie M5 baissière dans la zone\n"
    msg += "⏳ Invalidation : annuler après 12 bougies M15"
    return msg

last_signal = {"XAUUSD": {"direction": None, "price": None, "ts": 0}}

def should_send_signal(signal):
    prev    = last_signal["XAUUSD"]
    elapsed = time.time() - prev["ts"]
    if prev["direction"] != signal["direction"]: return True
    if elapsed >= SIGNAL_COOLDOWN:               return True
    if prev["price"] and abs(signal["price"]-prev["price"]) >= signal["atr"]*1.5: return True
    return False

async def main():
    bot = Bot(token=TELEGRAM_TOKEN,
              request=HTTPXRequest(read_timeout=30, connect_timeout=30, write_timeout=30))
    for attempt in range(5):
        try:
            await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=(
                "🚀 XauBot Signal v23.2 PRO démarré\n\n"
                "📊 M30 → tendance | M15 → structure | M5 → confirmation\n\n"
                "✅ Fixes v23.2 :\n"
                "• Break of Structure réparé\n"
                "• Plantage ordre limit SELL corrigé\n"
                f"• M5 score minimum {M5_MIN_SCORE}/80\n"
                "• Ordres SELL limit envoyés (rejet M5 en attente)\n"
                "• Fibonacci selon la direction\n"
                "• R:R sur vraie cible (swing)\n"
                "• Filtre news + ouverture NY (heure de New York)\n"
                "• EMA 200 M30 sur 500 bougies\n"
                f"🎯 Score minimum : {MIN_CONFIDENCE}/100"
            ))
            break
        except Exception as e:
            log.error("Startup: %s", e)
            await asyncio.sleep(10)

    log.info("XauBot v23.2 PRO démarré")

    while True:
        try:
            if not is_market_open():
                log.info("Marché fermé")
                await asyncio.sleep(SCAN_INTERVAL)
                continue

            if in_news_blackout():
                log.info("Fenêtre news / ouverture NY : pas de scan")
                await asyncio.sleep(SCAN_INTERVAL)
                continue

            signal = await analyze_xauusd()

            if signal and should_send_signal(signal):
                await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=format_message(signal))
                last_signal["XAUUSD"] = {"direction": signal["direction"],
                                          "price": signal["price"], "ts": time.time()}
                await asyncio.sleep(2)

                limit = calc_limit_entry(signal)
                if limit:
                    sell_rejected = False
                    if signal["direction"] == "SELL":
                        # FIX v23.2 : plus de "df or ..." sur un DataFrame
                        df5 = signal.get("m5_df5")
                        if df5 is None:
                            df5 = await get_candles(XAUUSD_CONFIG["symbol"], interval="5min", outputsize=20)
                        if df5 is not None:
                            sell_rejected = check_zone_rejection(df5, limit["limit"], "SELL")
                        # FIX v23.2 : plus d'annulation, l'ordre part avec "⏳ en attente"
                    limit_msg = format_limit_message(signal, limit, sell_rejected)
                    if limit_msg:
                        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=limit_msg)

        except Exception as e:
            log.exception("Erreur principale: %s", e)

        await asyncio.sleep(SCAN_INTERVAL)

if __name__ == "__main__":
    asyncio.run(main())
