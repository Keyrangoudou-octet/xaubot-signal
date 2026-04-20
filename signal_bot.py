# XauBot Signal Bot - Railway / Twelve Data
# Signaux XAUUSD + US100 sur M5 - Sessions Londres + New York uniquement

import asyncio
import logging
import os
import requests
import pandas as pd
from datetime import datetime, timezone
from telegram import Bot

TELEGRAM_TOKEN   = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
TWELVE_API_KEY   = os.environ["TWELVE_API_KEY"]

SCAN_INTERVAL = 300

# Sessions de trading (UTC)
SESSION_LONDON_START  = 8
SESSION_LONDON_END    = 17
SESSION_NY_START      = 13
SESSION_NY_END        = 22

XAUUSD_CONFIG = {
    "symbol"    : "XAU/USD",
    "label"     : "XAUUSD",
    "ema_fast"  : 15,
    "ema_slow"  : 50,
    "adx_period": 14,
    "adx_min"   : 20,
    "tp_pts"    : 1.5,
    "sl_pts"    : 1.0,
}

US100_CONFIG = {
    "symbol"    : "NDX",
    "label"     : "US100",
    "ema_fast"  : 20,
    "ema_slow"  : 50,
    "rsi_period": 14,
    "rsi_ob"    : 65,
    "rsi_os"    : 35,
    "tp_pts"    : 30,
    "sl_pts"    : 20,
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

def is_market_open():
    now_utc = datetime.now(timezone.utc)
    # Pas de trading le weekend
    if now_utc.weekday() >= 5:
        return False
    hour = now_utc.hour
    in_london = SESSION_LONDON_START <= hour < SESSION_LONDON_END
    in_ny     = SESSION_NY_START     <= hour < SESSION_NY_END
    return in_london or in_ny

def get_candles(symbol, interval="5min", outputsize=100):
    try:
        url = "https://api.twelvedata.com/time_series"
        params = {
            "symbol"    : symbol,
            "interval"  : interval,
            "outputsize": outputsize,
            "apikey"    : TWELVE_API_KEY,
            "format"    : "JSON"
        }
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        if "values" not in data:
            log.error("Twelve Data erreur " + symbol + ": " + str(data.get("message", "unknown")))
            return None
        df = pd.DataFrame(data["values"])
        df = df.rename(columns={"datetime": "time"})
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col])
        df = df.iloc[::-1].reset_index(drop=True)
        return df
    except Exception as e:
        log.error("get_candles " + symbol + ": " + str(e))
        return None

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))

def adx(df, period=14):
    high  = df["high"]
    low   = df["low"]
    close = df["close"]
    plus_dm  = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr      = tr.rolling(period).mean()
    plus_di  = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.rolling(period).mean(), plus_di, minus_di

def double_impulse(df):
    bull = (df["close"].iloc[-2] > df["open"].iloc[-2]) and (df["close"].iloc[-3] > df["open"].iloc[-3])
    bear = (df["close"].iloc[-2] < df["open"].iloc[-2]) and (df["close"].iloc[-3] < df["open"].iloc[-3])
    return bull, bear

def analyze_xauusd():
    cfg = XAUUSD_CONFIG
    df  = get_candles(cfg["symbol"])
    if df is None or len(df) < 60:
        return None
    df["ema_fast"] = ema(df["close"], cfg["ema_fast"])
    df["ema_slow"] = ema(df["close"], cfg["ema_slow"])
    adx_s, plus_di, minus_di = adx(df, cfg["adx_period"])
    price     = round(float(df["close"].iloc[-1]), 2)
    ema_f     = float(df["ema_fast"].iloc[-1])
    ema_s     = float(df["ema_slow"].iloc[-1])
    adx_now   = float(adx_s.iloc[-1])
    plus_now  = float(plus_di.iloc[-1])
    minus_now = float(minus_di.iloc[-1])
    bull_imp, bear_imp = double_impulse(df)
    adx_ok = adx_now > cfg["adx_min"]
    if ema_f > ema_s and plus_now > minus_now and adx_ok and bull_imp:
        return ("BUY",  price, round(price + cfg["tp_pts"], 2), round(price - cfg["sl_pts"], 2), round(adx_now, 1), "ADX")
    if ema_f < ema_s and minus_now > plus_now and adx_ok and bear_imp:
        return ("SELL", price, round(price - cfg["tp_pts"], 2), round(price + cfg["sl_pts"], 2), round(adx_now, 1), "ADX")
    return None

def analyze_us100():
    cfg = US100_CONFIG
    df  = get_candles(cfg["symbol"])
    if df is None or len(df) < 60:
        return None
    df["ema_fast"] = ema(df["close"], cfg["ema_fast"])
    df["ema_slow"] = ema(df["close"], cfg["ema_slow"])
    df["rsi"]      = rsi(df["close"], cfg["rsi_period"])
    price    = round(float(df["close"].iloc[-1]), 2)
    ema_f    = float(df["ema_fast"].iloc[-1])
    ema_s    = float(df["ema_slow"].iloc[-1])
    rsi_now  = float(df["rsi"].iloc[-1])
    rsi_prev = float(df["rsi"].iloc[-2])
    if ema_f > ema_s and rsi_prev < cfg["rsi_os"] and rsi_now > cfg["rsi_os"]:
        return ("BUY",  price, round(price + cfg["tp_pts"], 2), round(price - cfg["sl_pts"], 2), round(rsi_now, 1), "RSI")
    if ema_f < ema_s and rsi_prev > cfg["rsi_ob"] and rsi_now < cfg["rsi_ob"]:
        return ("SELL", price, round(price - cfg["tp_pts"], 2), round(price + cfg["sl_pts"], 2), round(rsi_now, 1), "RSI")
    return None

def format_message(label, direction, price, tp, sl, ind_val, ind_name):
    rr  = round(abs(tp - price) / abs(sl - price), 2)
    now = datetime.utcnow().strftime("%H:%M UTC")
    msg  = direction + " SIGNAL - " + label + "\n"
    msg += "Heure  : " + now + "\n"
    msg += "Entry  : " + str(price) + "\n"
    msg += "TP     : " + str(tp) + "\n"
    msg += "SL     : " + str(sl) + "\n"
    msg += "RR     : 1:" + str(rr) + "\n"
    msg += ind_name + "    : " + str(ind_val) + "\n"
    msg += "Signal indicatif - verifiez sur MT5"
    return msg

last_signal = {"XAUUSD": None, "US100": None}

async def main():
    bot = Bot(token=TELEGRAM_TOKEN)
    await bot.send_message(
        chat_id=TELEGRAM_CHAT_ID,
        text="XauBot Signal demarre - Sessions Londres + New York uniquement (8h-22h UTC)"
    )
    log.info("Bot demarre")

    while True:
        try:
            if not is_market_open():
                log.info("Marche ferme - attente")
                await asyncio.sleep(SCAN_INTERVAL)
                continue

            xau = analyze_xauusd()
            if xau:
                direction, price, tp, sl, ind, ind_name = xau
                key = direction + "_" + str(round(price, 0))
                if last_signal["XAUUSD"] != key:
                    msg = format_message("XAUUSD", direction, price, tp, sl, ind, ind_name)
                    await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)
                    last_signal["XAUUSD"] = key
                    log.info("Signal XAUUSD: " + direction + " @ " + str(price))
            else:
                last_signal["XAUUSD"] = None

            await asyncio.sleep(5)

            us = analyze_us100()
            if us:
                direction, price, tp, sl, ind, ind_name = us
                key = direction + "_" + str(round(price, 0))
                if last_signal["US100"] != key:
                    msg = format_message("US100", direction, price, tp, sl, ind, ind_name)
                    await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)
                    last_signal["US100"] = key
                    log.info("Signal US100: " + direction + " @ " + str(price))
            else:
                last_signal["US100"] = None

        except Exception as e:
            log.error("Erreur scan: " + str(e))

        await asyncio.sleep(SCAN_INTERVAL)

if __name__ == "__main__":
    asyncio.run(main())
