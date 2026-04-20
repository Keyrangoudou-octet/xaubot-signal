# XauBot Signal Bot - Railway / yfinance
# Signaux XAUUSD + US100 sur M5

import asyncio
import logging
import os
import pandas as pd
import yfinance as yf
from datetime import datetime
from telegram import Bot

TELEGRAM_TOKEN   = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

SCAN_INTERVAL = 300

XAUUSD_CONFIG = {
    "ticker"    : "GC=F",
    "label"     : "XAUUSD",
    "ema_fast"  : 15,
    "ema_slow"  : 50,
    "adx_period": 14,
    "adx_min"   : 20,
    "tp_pts"    : 1.5,
    "sl_pts"    : 1.0,
}

US100_CONFIG = {
    "ticker"    : "NQ=F",
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

def get_candles(ticker, period="2d", interval="5m"):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
        df = df.rename(columns={"open": "open", "high": "high", "low": "low", "close": "close"})
        df = df.dropna()
        return df
    except Exception as e:
        log.error("get_candles " + ticker + ": " + str(e))
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
    df  = get_candles(cfg["ticker"])
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
    df  = get_candles(cfg["ticker"])
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
    arrow = "BUY" if direction == "BUY" else "SELL"
    msg  = arrow + " SIGNAL " + direction + " - " + label + "\n"
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
        text="XauBot Signal demarre - Scan XAUUSD + US100 toutes les 5 min"
    )
    log.info("Bot demarre")

    while True:
        try:
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
