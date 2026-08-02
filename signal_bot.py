# XauBot Signal Bot - Railway / Twelve Data
# v16 : FVG 15M + Order Block 15M | Confluence FVG+OB

import asyncio, logging, os, time, requests, pandas as pd
from datetime import datetime, timezone
from telegram import Bot

TELEGRAM_TOKEN   = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
TWELVE_API_KEY   = os.environ["TWELVE_API_KEY"]

SCAN_INTERVAL = 180    # 3 min — optimise pour quota 800 calls/jour (3 calls/scan : 30M + 5M + 15M FVG)
SIGNAL_COOLDOWN = 900  # 15 min entre deux signaux meme direction

XAUUSD_CONFIG = {
    "symbol": "XAU/USD", "label": "XAUUSD",
    "ema_fast": 15, "ema_slow": 50,
    "adx_period": 14, "adx_min": 25,
    "atr_period": 14, "atr_sl_mult": 1.5,
    "fibo_lookback": 50,
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

def is_market_open():
    now_utc = datetime.now(timezone.utc)
    wd = now_utc.weekday()
    h  = now_utc.hour
    if wd == 5: return False
    if wd == 6: return False
    if wd == 4: return 8 <= h < 17
    return 8 <= h < 19

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

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def recommend_fibo(rsi_v, adx_v):
    if rsi_v > 70 and adx_v > 35:
        return "38.2"
    elif rsi_v >= 65 or adx_v >= 30:
        return "50.0"
    else:
        return "61.8"

def detect_fvg(df, lookback=30):
    price = float(df["close"].iloc[-1])
    n = len(df)
    end = n - 1
    for i in range(end - 1, max(2, end - lookback), -1):
        h0 = float(df["high"].iloc[i - 2])
        l0 = float(df["low"].iloc[i - 2])
        hi = float(df["high"].iloc[i])
        li = float(df["low"].iloc[i])
        if li > h0 and h0 <= price <= li:
            return ("BULL", round(h0, 2), round(li, 2))
        if hi < l0 and hi <= price <= l0:
            return ("BEAR", round(hi, 2), round(l0, 2))
    return None

def detect_ob(df, lookback=30, atr_mult=1.5):
    price = float(df["close"].iloc[-1])
    n = len(df)
    atr_v = float(atr(df).iloc[-1])
    end = n - 1
    for i in range(max(1, end - lookback), end - 3):
        o  = float(df["open"].iloc[i])
        c  = float(df["close"].iloc[i])
        h  = float(df["high"].iloc[i])
        l  = float(df["low"].iloc[i])
        fu_h = max(float(df["high"].iloc[j]) for j in range(i+1, min(i+4, n)))
        fu_l = min(float(df["low"].iloc[j])  for j in range(i+1, min(i+4, n)))
        if c < o and (fu_h - h) > atr_mult * atr_v:
            if l <= price <= h:
                return ("BULL", round(l, 2), round(h, 2))
        if c > o and (l - fu_l) > atr_mult * atr_v:
            if l <= price <= h:
                return ("BEAR", round(l, 2), round(h, 2))
    return None

def double_impulse(df):
    bull = (df["close"].iloc[-2]>df["open"].iloc[-2]) and (df["close"].iloc[-3]>df["open"].iloc[-3])
    bear = (df["close"].iloc[-2]<df["open"].iloc[-2]) and (df["close"].iloc[-3]<df["open"].iloc[-3])
    return bull, bear

def live_breakout(df, atr_v):
    c1 = float(df["close"].iloc[-1]); o1 = float(df["open"].iloc[-1])
    c2 = float(df["close"].iloc[-2]); o2 = float(df["open"].iloc[-2])
    body1 = abs(c1 - o1); body2 = abs(c2 - o2)
    if (c1 > o1) and (body1 > atr_v * 2.0): return True, False, False
    if (c1 < o1) and (body1 > atr_v * 2.0): return False, True, False
    if (c2 > o2) and (body2 > atr_v * 2.0) and (c1 > o1) and (body1 > atr_v * 0.3): return True, False, True
    if (c2 < o2) and (body2 > atr_v * 2.0) and (c1 < o1) and (body1 > atr_v * 0.3): return False, True, True
    return False, False, False

def detect_pattern(df):
    o  = float(df["open"].iloc[-2]);  h = float(df["high"].iloc[-2])
    l  = float(df["low"].iloc[-2]);   c = float(df["close"].iloc[-2])
    o2 = float(df["open"].iloc[-3]);  c2 = float(df["close"].iloc[-3])
    total = h - l
    if total == 0: return None
    body = abs(c - o)
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    bp = body / total; up = upper_wick / total; lp = lower_wick / total
    if bp < 0.1:                           return "Doji"
    if bp > 0.85:                          return "Marubozu " + ("Bull" if c > o else "Bear")
    if lp > 0.6 and bp < 0.3:             return "Pin Bar Bull"
    if up > 0.6 and bp < 0.3:             return "Pin Bar Bear"
    if c > o and c2 < o2 and c > o2 and o < c2: return "Engulfing Bull"
    if c < o and c2 > o2 and c < o2 and o > c2: return "Engulfing Bear"
    return None

def fibo_range(df, lookback=50):
    r = df.tail(lookback)
    return round(float(r["high"].max()), 2), round(float(r["low"].min()), 2)

def fibonacci_levels(sh, sl):
    d = sh - sl
    if d == 0: return None
    return {"38.2": round(sh-0.382*d,2), "50.0": round(sh-0.500*d,2), "61.8": round(sh-0.618*d,2)}

def nearest_fibo(price, fib):
    if fib is None: return "---"
    best, dist = None, float("inf")
    for k in ["38.2", "50.0", "61.8"]:
        d = abs(price - fib[k])
        if d < dist: dist, best = d, k
    return best

def get_htf_trend(symbol):
    df = get_candles(symbol, interval="30min", outputsize=60)
    if df is None or len(df) < 55: return None
    df["ef"] = ema(df["close"], 15); df["es"] = ema(df["close"], 50)
    return "BULL" if float(df["ef"].iloc[-1]) > float(df["es"].iloc[-1]) else "BEAR"

def analyze_xauusd():
    cfg = XAUUSD_CONFIG
    htf = get_htf_trend(cfg["symbol"])
    if not htf: return None
    df = get_candles(cfg["symbol"])
    if df is None or len(df) < 60: return None
    df["ef"] = ema(df["close"], cfg["ema_fast"]); df["es"] = ema(df["close"], cfg["ema_slow"])
    df["atr_v"] = atr(df, cfg["atr_period"])
    adx_s, pdi, mdi = adx(df, cfg["adx_period"])
    rsi_s  = rsi(df["close"], 14)
    price  = round(float(df["close"].iloc[-1]), 2)
    ef, es = float(df["ef"].iloc[-1]), float(df["es"].iloc[-1])
    adx_v  = float(adx_s.iloc[-1]); pdi_v = float(pdi.iloc[-1]); mdi_v = float(mdi.iloc[-1])
    rsi_v  = round(float(rsi_s.iloc[-1]), 1)
    atr_v  = float(df["atr_v"].iloc[-1])
    bull_i, bear_i = double_impulse(df)
    bull_live, bear_live, confirmed = live_breakout(df, atr_v)
    sh, sl_s = fibo_range(df, cfg["fibo_lookback"])
    fib = fibonacci_levels(sh, sl_s)
    fib_lvl = nearest_fibo(price, fib)
    pattern = detect_pattern(df)
    df15    = get_candles(cfg["symbol"], interval="15min", outputsize=60)
    fvg     = detect_fvg(df15) if df15 is not None and len(df15) >= 10 else None
    ob      = detect_ob(df15)  if df15 is not None and len(df15) >= 10 else None
    sd = round(atr_v * cfg["atr_sl_mult"], 2)
    tp1_d = round(atr_v * 0.8, 2)
    tp2_d = round(atr_v * 1.5, 2)
    tp3_d = round(atr_v * 2.5, 2)

    if pdi_v>mdi_v and adx_v>cfg["adx_min"] and bull_live:
        st = "BREAKOUT_CONF" if confirmed else "BREAKOUT"
        return ("BUY", price, round(price-sd,2), round(price+tp1_d,2), round(price+tp2_d,2), round(price+tp3_d,2), round(adx_v,1), htf, fib, fib_lvl, st, pattern, rsi_v, fvg, ob)
    if mdi_v>pdi_v and adx_v>cfg["adx_min"] and bear_live:
        st = "BREAKOUT_CONF" if confirmed else "BREAKOUT"
        return ("SELL",price, round(price+sd,2), round(price-tp1_d,2), round(price-tp2_d,2), round(price-tp3_d,2), round(adx_v,1), htf, fib, fib_lvl, st, pattern, rsi_v, fvg, ob)
    if ef>es and pdi_v>mdi_v and adx_v>cfg["adx_min"] and bull_i and htf=="BULL":
        return ("BUY", price, round(price-sd,2), round(price+tp1_d,2), round(price+tp2_d,2), round(price+tp3_d,2), round(adx_v,1), htf, fib, fib_lvl, "SIGNAL", pattern, rsi_v, fvg, ob)
    if ef<es and mdi_v>pdi_v and adx_v>cfg["adx_min"] and bear_i and htf=="BEAR":
        return ("SELL",price, round(price+sd,2), round(price-tp1_d,2), round(price-tp2_d,2), round(price-tp3_d,2), round(adx_v,1), htf, fib, fib_lvl, "SIGNAL", pattern, rsi_v, fvg, ob)
    return None

def format_message(label, direction, price, sl, tp1, tp2, tp3, val, htf, fib, fib_lvl, signal_type="SIGNAL", pattern=None, rsi_v=None, fvg=None, ob=None):
    now   = datetime.utcnow().strftime("%H:%M UTC")
    arrow = "\U0001f7e2" if direction == "BUY" else "\U0001f534"
    icon  = "✅" if (direction=="BUY" and htf=="BULL") or (direction=="SELL" and htf=="BEAR") else "⚠️"
    sl_d  = round(abs(price - sl), 2)

    if signal_type in ("BREAKOUT", "BREAKOUT_CONF"):
        msg  = "⚡ BREAKOUT " + direction + " - " + label + "\n"
        msg += "━━━━━━━━━━━━━━━━━━\n"
        msg += "\U0001f525 Rupture en cours — preparer le retracement\n"
        msg += "━━━━━━━━━━━━━━━━━━\n"
    else:
        msg  = arrow + " " + direction + " SIGNAL - " + label + "\n"
        msg += "━━━━━━━━━━━━━━━━━━\n"

    msg += "\U0001f550 Heure  : " + now + "\n"
    msg += "\U0001f4cd Entry  : " + str(price) + "\n"
    msg += "\U0001f6d1 SL     : " + str(sl) + "  (-" + str(sl_d) + ")\n"
    msg += "━━━━━━━━━━━━━━━━━━\n"
    msg += "\U0001f3af TP1    : " + str(tp1) + "  (securiser 50%)\n"
    msg += "\U0001f3af TP2    : " + str(tp2) + "  (RR 1:1)\n"
    msg += "\U0001f3af TP3    : " + str(tp3) + "  (RR 1:1.7)\n"
    msg += "━━━━━━━━━━━━━━━━━━\n"
    msg += "\U0001f4ca ADX    : " + str(val) + "\n"
    if rsi_v is not None:
        msg += "\U0001f4c9 RSI    : " + str(rsi_v) + "\n"
    if fvg:
        fvg_icon = "✅" if (fvg[0]=="BULL" and direction=="BUY") or (fvg[0]=="BEAR" and direction=="SELL") else "⚠️"
        msg += "\U0001f4d0 FVG 15M : " + fvg[0] + " " + fvg_icon + " [" + str(fvg[1]) + "-" + str(fvg[2]) + "]\n"
    else:
        msg += "\U0001f4d0 FVG 15M : -\n"
    if ob:
        ob_icon = "✅" if (ob[0]=="BULL" and direction=="BUY") or (ob[0]=="BEAR" and direction=="SELL") else "⚠️"
        msg += "\U0001f4e6 OB 15M  : " + ob[0] + " " + ob_icon + " [" + str(ob[1]) + "-" + str(ob[2]) + "]\n"
    else:
        msg += "\U0001f4e6 OB 15M  : -\n"
    if fvg and ob and fvg[0] == ob[0] and ((fvg[0]=="BULL" and direction=="BUY") or (fvg[0]=="BEAR" and direction=="SELL")):
        msg += "\U0001f525 Confluence FVG+OB — zone tres forte !\n"
    msg += "\U0001f4c8 M30    : " + htf + " " + icon + "\n"
    if pattern:
        msg += "\U0001f56f Pattern : " + pattern + "\n"

    if signal_type in ("BREAKOUT", "BREAKOUT_CONF"):
        msg += "━━━━━━━━━━━━━━━━━━\n"
        if fib:
            rec = recommend_fibo(rsi_v, val) if rsi_v is not None else "50.0"
            for lvl in ["38.2", "50.0", "61.8"]:
                star = " ⭐ RECOMMANDE" if lvl == rec else ""
                msg += "\U0001f4d0 Fibo " + lvl + "% : " + str(fib[lvl]) + star + "\n"
        msg += "━━━━━━━━━━━━━━━━━━\n"
        if signal_type == "BREAKOUT_CONF":
            msg += "✅ Breakout confirme — attends le retracement Fibo"
        else:
            msg += "⚠️ Bougie en cours — attends la cloture et le retracement"
    else:
        msg += "━━━━━━━━━━━━━━━━━━\n"
        if fib:
            rec = recommend_fibo(rsi_v, val) if rsi_v is not None else "50.0"
            for lvl in ["38.2", "50.0", "61.8"]:
                star = " ⭐ RECOMMANDE" if lvl == rec else ""
                msg += "\U0001f4d0 Fibo " + lvl + "% : " + str(fib[lvl]) + star + "\n"
        msg += "━━━━━━━━━━━━━━━━━━\n"
        msg += "📌 Placez votre limit order au Fibo ⭐"
    return msg

last_signal = {"XAUUSD": {"direction": None, "type": None, "ts": 0}}

async def main():
    bot = Bot(token=TELEGRAM_TOKEN)
    await bot.send_message(chat_id=TELEGRAM_CHAT_ID,
        text="XauBot Signal v16 demarre\nScan 3min | Lun-Jeu 8h-19h UTC | Ven 8h-17h UTC | ADX 25 | RSI + Fibo | FVG 15M | OB 15M | Confluence | M30 | Cooldown 15min")
    log.info("Bot demarre v16")
    while True:
        try:
            if not is_market_open():
                log.info("Marche ferme"); await asyncio.sleep(SCAN_INTERVAL); continue
            xau = analyze_xauusd()
            if xau:
                d,p,sl,tp1,tp2,tp3,v,htf,fib,fl,st,pat,rsiv,fvg,ob = xau
                prev = last_signal["XAUUSD"]
                elapsed = time.time() - prev["ts"]
                same = (prev["direction"] == d and prev["type"] == st)
                if not same or elapsed > SIGNAL_COOLDOWN:
                    await bot.send_message(chat_id=TELEGRAM_CHAT_ID,
                        text=format_message("XAUUSD",d,p,sl,tp1,tp2,tp3,v,htf,fib,fl,st,pat,rsiv,fvg,ob))
                    last_signal["XAUUSD"] = {"direction": d, "type": st, "ts": time.time()}
                    fvg_log = (fvg[0]+" ["+str(fvg[1])+"-"+str(fvg[2])+"]") if fvg else "no FVG"
                    log.info("XAUUSD "+st+" "+d+" @ "+str(p)+" | "+htf+" | RSI "+str(rsiv)+" | Fibo "+str(fl)+" | FVG "+fvg_log+" | "+str(pat))
        except Exception as e:
            log.error("Erreur: "+str(e))
        await asyncio.sleep(SCAN_INTERVAL)

if __name__ == "__main__":
    asyncio.run(main())
