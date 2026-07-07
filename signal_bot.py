# XauBot Signal Bot - Railway / Twelve Data
# v3 : filtre H1 + ATR + 3 TP + Fibonacci swing points

import asyncio, logging, os, requests, pandas as pd
from datetime import datetime, timezone
from telegram import Bot

TELEGRAM_TOKEN   = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
TWELVE_API_KEY   = os.environ["TWELVE_API_KEY"]

SCAN_INTERVAL = 300
SESSION_LONDON_START, SESSION_LONDON_END = 8, 17
SESSION_NY_START, SESSION_NY_END = 13, 22

XAUUSD_CONFIG = {
    "symbol": "XAU/USD", "label": "XAUUSD",
    "ema_fast": 15, "ema_slow": 50,
    "adx_period": 14, "adx_min": 20,
    "atr_period": 14, "atr_sl_mult": 1.5,
    "swing_window": 5, "swing_lookback": 100, "fibo_atr_mult": 0.75,
}
US100_CONFIG = {
    "symbol": "NDX", "label": "US100",
    "ema_fast": 20, "ema_slow": 50,
    "rsi_period": 14, "rsi_ob": 65, "rsi_os": 35,
    "atr_period": 14, "atr_sl_mult": 1.5,
    "swing_window": 5, "swing_lookback": 100, "fibo_atr_mult": 0.75,
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

def is_market_open():
    now_utc = datetime.now(timezone.utc)
    if now_utc.weekday() >= 5: return False
    h = now_utc.hour
    return (SESSION_LONDON_START <= h < SESSION_LONDON_END) or (SESSION_NY_START <= h < SESSION_NY_END)

def get_candles(symbol, interval="5min", outputsize=120):
    try:
        r = requests.get("https://api.twelvedata.com/time_series", params={
            "symbol": symbol, "interval": interval,
            "outputsize": outputsize, "apikey": TWELVE_API_KEY, "format": "JSON"
        }, timeout=10)
        data = r.json()
        if "values" not in data:
            log.error("Twelve Data erreur " + symbol + ": " + str(data.get("message", "")))
            return None
        df = pd.DataFrame(data["values"]).rename(columns={"datetime": "time"})
        for col in ["open", "high", "low", "close"]: df[col] = pd.to_numeric(df[col])
        return df.iloc[::-1].reset_index(drop=True)
    except Exception as e:
        log.error("get_candles " + symbol + ": " + str(e)); return None

def ema(series, period): return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    d = series.diff()
    return 100 - (100 / (1 + d.clip(lower=0).rolling(period).mean() / (-d.clip(upper=0)).rolling(period).mean()))

def adx(df, period=14):
    hi, lo, cl = df["high"], df["low"], df["close"]
    plus_dm = hi.diff().clip(lower=0); minus_dm = (-lo.diff()).clip(lower=0)
    tr = pd.concat([hi-lo, (hi-cl.shift()).abs(), (lo-cl.shift()).abs()], axis=1).max(axis=1)
    ar = tr.rolling(period).mean()
    pdi = 100*(plus_dm.rolling(period).mean()/ar); mdi = 100*(minus_dm.rolling(period).mean()/ar)
    return (100*(pdi-mdi).abs()/(pdi+mdi)).rolling(period).mean(), pdi, mdi

def atr(df, period=14):
    hi, lo, cl = df["high"], df["low"], df["close"]
    return pd.concat([hi-lo, (hi-cl.shift()).abs(), (lo-cl.shift()).abs()], axis=1).max(axis=1).rolling(period).mean()

def double_impulse(df):
    bull = (df["close"].iloc[-2]>df["open"].iloc[-2]) and (df["close"].iloc[-3]>df["open"].iloc[-3])
    bear = (df["close"].iloc[-2]<df["open"].iloc[-2]) and (df["close"].iloc[-3]<df["open"].iloc[-3])
    return bull, bear

# ── FIBONACCI SWING POINTS ──────────────────────────────

def find_swing_high(df, window=5, lookback=100):
    """Dernier swing high : high supérieur aux `window` bougies de chaque côté."""
    r = df.tail(lookback).reset_index(drop=True)
    for i in range(len(r)-window-1, window-1, -1):
        hi = r["high"].iloc[i]
        if all(hi >= r["high"].iloc[i-j] for j in range(1, window+1)) and \
           all(hi >= r["high"].iloc[i+j] for j in range(1, window+1)):
            return round(float(hi), 2)
    return round(float(r["high"].max()), 2)

def find_swing_low(df, window=5, lookback=100):
    """Dernier swing low : low inférieur aux `window` bougies de chaque côté."""
    r = df.tail(lookback).reset_index(drop=True)
    for i in range(len(r)-window-1, window-1, -1):
        lo = r["low"].iloc[i]
        if all(lo <= r["low"].iloc[i-j] for j in range(1, window+1)) and \
           all(lo <= r["low"].iloc[i+j] for j in range(1, window+1)):
            return round(float(lo), 2)
    return round(float(r["low"].min()), 2)

def fibonacci_levels(sh, sl):
    d = sh - sl
    if d == 0: return None
    return {"38.2": round(sh-0.382*d,2), "50.0": round(sh-0.500*d,2), "61.8": round(sh-0.618*d,2)}

def near_fibo_level(price, fib, atr_val, mult=0.75):
    if fib is None: return None
    tol = atr_val * mult
    best, dist = None, float("inf")
    for k in ["38.2", "50.0", "61.8"]:
        d = abs(price - fib[k])
        if d <= tol and d < dist: dist, best = d, k
    return best

# ── FILTRE H1 ───────────────────────────────────────────

def get_htf_trend(symbol):
    df = get_candles(symbol, interval="1h", outputsize=60)
    if df is None or len(df) < 55: return None
    df["ef"] = ema(df["close"], 15); df["es"] = ema(df["close"], 50)
    return "BULL" if float(df["ef"].iloc[-1]) > float(df["es"].iloc[-1]) else "BEAR"

# ── XAUUSD ──────────────────────────────────────────────

def analyze_xauusd():
    cfg = XAUUSD_CONFIG
    htf = get_htf_trend(cfg["symbol"])
    if not htf: return None
    df = get_candles(cfg["symbol"])
    if df is None or len(df) < 60: return None
    df["ef"] = ema(df["close"], cfg["ema_fast"]); df["es"] = ema(df["close"], cfg["ema_slow"])
    df["atr_v"] = atr(df, cfg["atr_period"])
    adx_s, pdi, mdi = adx(df, cfg["adx_period"])
    price  = round(float(df["close"].iloc[-1]), 2)
    ef, es = float(df["ef"].iloc[-1]), float(df["es"].iloc[-1])
    adx_v  = float(adx_s.iloc[-1]); pdi_v = float(pdi.iloc[-1]); mdi_v = float(mdi.iloc[-1])
    atr_v  = float(df["atr_v"].iloc[-1])
    bull_i, bear_i = double_impulse(df)
    # Fibonacci
    sh = find_swing_high(df, cfg["swing_window"], cfg["swing_lookback"])
    sl = find_swing_low(df,  cfg["swing_window"], cfg["swing_lookback"])
    fib = fibonacci_levels(sh, sl)
    fib_lvl = near_fibo_level(price, fib, atr_v, cfg["fibo_atr_mult"])
    if not fib_lvl:
        log.info("XAUUSD hors zone Fibo - ignoré"); return None
    sd = round(atr_v * cfg["atr_sl_mult"], 2)
    if ef>es and pdi_v>mdi_v and adx_v>cfg["adx_min"] and bull_i and htf=="BULL":
        sl_p = round(price-sd,2)
        return ("BUY", price, sl_p, round(price+sd,2), round(price+sd*2,2), round(price+sd*3,2), round(adx_v,1), htf, fib_lvl)
    if ef<es and mdi_v>pdi_v and adx_v>cfg["adx_min"] and bear_i and htf=="BEAR":
        sl_p = round(price+sd,2)
        return ("SELL",price, sl_p, round(price-sd,2), round(price-sd*2,2), round(price-sd*3,2), round(adx_v,1), htf, fib_lvl)
    return None

# ── US100 ───────────────────────────────────────────────

def analyze_us100():
    cfg = US100_CONFIG
    htf = get_htf_trend(cfg["symbol"])
    if not htf: return None
    df = get_candles(cfg["symbol"])
    if df is None or len(df) < 60: return None
    df["ef"] = ema(df["close"], cfg["ema_fast"]); df["es"] = ema(df["close"], cfg["ema_slow"])
    df["rsi_v"] = rsi(df["close"], cfg["rsi_period"]); df["atr_v"] = atr(df, cfg["atr_period"])
    price   = round(float(df["close"].iloc[-1]), 2)
    ef, es  = float(df["ef"].iloc[-1]), float(df["es"].iloc[-1])
    rsi_now = float(df["rsi_v"].iloc[-1]); rsi_prev = float(df["rsi_v"].iloc[-2])
    atr_v   = float(df["atr_v"].iloc[-1])
    sh = find_swing_high(df, cfg["swing_window"], cfg["swing_lookback"])
    sl = find_swing_low(df,  cfg["swing_window"], cfg["swing_lookback"])
    fib = fibonacci_levels(sh, sl)
    fib_lvl = near_fibo_level(price, fib, atr_v, cfg["fibo_atr_mult"])
    if not fib_lvl:
        log.info("US100 hors zone Fibo - ignoré"); return None
    sd = round(atr_v * cfg["atr_sl_mult"], 2)
    if ef>es and rsi_prev<cfg["rsi_os"] and rsi_now>cfg["rsi_os"] and htf=="BULL":
        sl_p = round(price-sd,2)
        return ("BUY", price, sl_p, round(price+sd,2), round(price+sd*2,2), round(price+sd*3,2), round(rsi_now,1), htf, fib_lvl)
    if ef<es and rsi_prev>cfg["rsi_ob"] and rsi_now<cfg["rsi_ob"] and htf=="BEAR":
        sl_p = round(price+sd,2)
        return ("SELL",price, sl_p, round(price-sd,2), round(price-sd*2,2), round(price-sd*3,2), round(rsi_now,1), htf, fib_lvl)
    return None

# ── FORMAT MESSAGE ───────────────────────────────────────

def format_message(label, direction, price, sl, tp1, tp2, tp3, val, htf, fib_level):
    now = datetime.utcnow().strftime("%H:%M UTC")
    arrow = "🟢" if direction == "BUY" else "🔴"
    icon  = "✅" if (direction=="BUY" and htf=="BULL") or (direction=="SELL" and htf=="BEAR") else "⚠️"
    sl_d  = round(abs(price-sl), 2)
    msg  = arrow + " " + direction + " SIGNAL - " + label + "\n"
    msg += "━━━━━━━━━━━━━━━━━━\n"
    msg += "🕐 Heure  : " + now + "\n"
    msg += "📍 Entry  : " + str(price) + "\n"
    msg += "🛑 SL     : " + str(sl) + "  (-" + str(sl_d) + ")\n"
    msg += "━━━━━━━━━━━━━━━━━━\n"
    msg += "🎯 TP1    : " + str(tp1) + "  (RR 1:1)\n"
    msg += "🎯 TP2    : " + str(tp2) + "  (RR 1:2)\n"
    msg += "🎯 TP3    : " + str(tp3) + "  (RR 1:3)\n"
    msg += "━━━━━━━━━━━━━━━━━━\n"
    msg += "📊 ADX    : " + str(val) + "\n"
    msg += "📈 H1     : " + htf + " " + icon + "\n"
    msg += "📐 Fibo   : " + str(fib_level) + "% ✅\n"
    msg += "━━━━━━━━━━━━━━━━━━\n"
    msg += "⚠️ Signal indicatif - vérifiez sur MT5"
    return msg

# ── MAIN ────────────────────────────────────────────────

last_signal = {"XAUUSD": None, "US100": None}

async def main():
    bot = Bot(token=TELEGRAM_TOKEN)
    await bot.send_message(chat_id=TELEGRAM_CHAT_ID,
        text="🤖 XauBot Signal v3 démarré\nFiltre H1 ✅ | Fibonacci swing points 38.2/50/61.8% ✅")
    log.info("Bot démarré v3")
    while True:
        try:
            if not is_market_open():
                log.info("Marché fermé"); await asyncio.sleep(SCAN_INTERVAL); continue
            xau = analyze_xauusd()
            if xau:
                d,p,sl,tp1,tp2,tp3,v,htf,fl = xau
                key = d+"_"+str(round(p,0))
                if last_signal["XAUUSD"] != key:
                    await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=format_message("XAUUSD",d,p,sl,tp1,tp2,tp3,v,htf,fl))
                    last_signal["XAUUSD"] = key
                    log.info("XAUUSD "+d+" @ "+str(p)+" | Fibo "+fl+"% | "+htf)
            else: last_signal["XAUUSD"] = None
            await asyncio.sleep(5)
            us = analyze_us100()
            if us:
                d,p,sl,tp1,tp2,tp3,v,htf,fl = us
                key = d+"_"+str(round(p,0))
                if last_signal["US100"] != key:
                    await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=format_message("US100",d,p,sl,tp1,tp2,tp3,v,htf,fl))
                    last_signal["US100"] = key
                    log.info("US100 "+d+" @ "+str(p)+" | Fibo "+fl+"% | "+htf)
            else: last_signal["US100"] = None
        except Exception as e:
            log.error("Erreur: "+str(e))
        await asyncio.sleep(SCAN_INTERVAL)

if __name__ == "__main__":
    asyncio.run(main())
