
XauBot - Signal Bot Telegram (Railway / MetaAPI)
Signaux XAUUSD + US100 sur M5


import asyncio
import logging
import os
import pandas as pd
from datetime import datetime
from telegram import Bot
from telegram.constants import ParseMode
from metaapi_cloud_sdk import MetaApi

# ─────────────────────────────────────────────

# CONFIG — variables d’environnement Railway

# ─────────────────────────────────────────────

TELEGRAM_TOKEN   = os.environ[“TELEGRAM_TOKEN”]
TELEGRAM_CHAT_ID = os.environ[“TELEGRAM_CHAT_ID”]
METAAPI_TOKEN    = os.environ[“METAAPI_TOKEN”]
METAAPI_ACCOUNT  = os.environ[“METAAPI_ACCOUNT”]

SCAN_INTERVAL = 300  # 5 minutes

# ─────────────────────────────────────────────

# PARAMÈTRES SIGNAUX

# ─────────────────────────────────────────────

XAUUSD_CONFIG = {
"symbol”    : “XAUUSD”,
"ema_fast”  : 15,
"ema_slow”  : 50,
"adx_period”: 14,
"adx_min”   : 20,
"tp_pts”    : 1.5,
"sl_pts”    : 1.0,
}

US100_CONFIG = {
"symbol”    : “US100.cash”,
"ema_fast”  : 20,
"ema_slow”  : 50,
"rsi_period”: 14,
"rsi_ob”    : 65,
"rsi_os”    : 35,
"tp_pts”    : 30,
"sl_pts”    : 20,
}

# ─────────────────────────────────────────────

# LOGGING

# ─────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format=”%(asctime)s | %(levelname)s | %(message)s”)
log = logging.getLogger(**name**)

# ─────────────────────────────────────────────

# INDICATEURS

# ─────────────────────────────────────────────

def ema(series, period):
return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
delta = series.diff()
gain  = delta.clip(lower=0).rolling(period).mean()
loss  = (-delta.clip(upper=0)).rolling(period).mean()
rs    = gain / loss
return 100 - (100 / (1 + rs))

def adx(df, period=14):
high, low, close = df[“high”], df[“low”], df[“close”]
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
bull = (df[“close”].iloc[-2] > df[“open”].iloc[-2]) and (df[“close”].iloc[-3] > df[“open”].iloc[-3])
bear = (df[“close”].iloc[-2] < df[“open”].iloc[-2]) and (df[“close”].iloc[-3] < df[“open”].iloc[-3])
return bull, bear

# ─────────────────────────────────────────────

# BOUGIES via MetaAPI

# ─────────────────────────────────────────────

async def get_candles(connection, symbol, count=200):
try:
candles = await connection.get_historical_candles(symbol, “5m”, count)
df = pd.DataFrame([{“time”: c[“time”], “open”: c[“open”],
“high”: c[“high”], “low”: c[“low”], “close”: c[“close”]}
for c in candles])
return df
except Exception as e:
log.error(f”get_candles {symbol}: {e}”)
return None

# ─────────────────────────────────────────────

# ANALYSE

# ─────────────────────────────────────────────

async def analyze_xauusd(connection):
cfg = XAUUSD_CONFIG
df  = await get_candles(connection, cfg[“symbol”])
if df is None or len(df) < 60:
return None
df[“ema_fast”] = ema(df[“close”], cfg[“ema_fast”])
df[“ema_slow”] = ema(df[“close”], cfg[“ema_slow”])
adx_s, plus_di, minus_di = adx(df, cfg[“adx_period”])
price, ema_f, ema_s = df[“close”].iloc[-1], df[“ema_fast”].iloc[-1], df[“ema_slow”].iloc[-1]
adx_now, plus_now, minus_now = adx_s.iloc[-1], plus_di.iloc[-1], minus_di.iloc[-1]
bull_imp, bear_imp = double_impulse(df)
adx_ok = adx_now > cfg[“adx_min”]
if ema_f > ema_s and plus_now > minus_now and adx_ok and bull_imp:
return (“BUY”,  price, round(price + cfg[“tp_pts”], 2), round(price - cfg[“sl_pts”], 2), round(adx_now, 1), “ADX”)
if ema_f < ema_s and minus_now > plus_now and adx_ok and bear_imp:
return (“SELL”, price, round(price - cfg[“tp_pts”], 2), round(price + cfg[“sl_pts”], 2), round(adx_now, 1), “ADX”)
return None

async def analyze_us100(connection):
cfg = US100_CONFIG
df  = await get_candles(connection, cfg[“symbol”])
if df is None or len(df) < 60:
return None
df[“ema_fast”] = ema(df[“close”], cfg[“ema_fast”])
df[“ema_slow”] = ema(df[“close”], cfg[“ema_slow”])
df[“rsi”]      = rsi(df[“close”], cfg[“rsi_period”])
price, ema_f, ema_s = df[“close”].iloc[-1], df[“ema_fast”].iloc[-1], df[“ema_slow”].iloc[-1]
rsi_now, rsi_prev = df[“rsi”].iloc[-1], df[“rsi”].iloc[-2]
if ema_f > ema_s and rsi_prev < cfg[“rsi_os”] and rsi_now > cfg[“rsi_os”]:
return (“BUY”,  price, round(price + cfg[“tp_pts”], 2), round(price - cfg[“sl_pts”], 2), round(rsi_now, 1), “RSI”)
if ema_f < ema_s and rsi_prev > cfg[“rsi_ob”] and rsi_now < cfg[“rsi_ob”]:
return (“SELL”, price, round(price - cfg[“tp_pts”], 2), round(price + cfg[“sl_pts”], 2), round(rsi_now, 1), “RSI”)
return None

# ─────────────────────────────────────────────

# MESSAGE

# ─────────────────────────────────────────────

def format_message(symbol, direction, price, tp, sl, ind_val, ind_name):
emoji = “🟢” if direction == “BUY” else “🔴”
rr    = round(abs(tp - price) / abs(sl - price), 2)
now   = datetime.utcnow().strftime(”%H:%M UTC”)
return (
f"{emoji} *SIGNAL {direction} — {symbol}*\n"
f”━━━━━━━━━━━━━━━━━━\n"
f"⏰ *Heure :* `{now}`\n"
f"💰 *Entry :* `{price}`\n"
f"✅ *TP :* `{tp}`\n"
f"🛑 *SL :* `{sl}`\n"
f"📊 *RR :* `1:{rr}`\n"
f"📈 *{ind_name} :* `{ind_val}`\n"
f"━━━━━━━━━━━━━━━━━━\n”
f"⚠️ *Signal indicatif — gérez votre risque*"
)

# ─────────────────────────────────────────────

# BOUCLE

# ─────────────────────────────────────────────

last_signal = {“XAUUSD”: None, "US100.cash": None}

async def main():
bot = Bot(token=TELEGRAM_TOKEN)
api = MetaApi(METAAPI_TOKEN)

```
log.info("Connexion MetaAPI...")
account    = await api.metatrader_account_api.get_account(METAAPI_ACCOUNT)
connection = account.get_rpc_connection()
await connection.connect()
await connection.wait_synchronized()
log.info("MetaAPI connecté")

await bot.send_message(
    chat_id=TELEGRAM_CHAT_ID,
    text="🤖 *XauBot Signal* démarré\nScan XAUUSD + US100 toutes les 5 min (M5)",
    parse_mode=ParseMode.MARKDOWN
)

while True:
    try:
        for analyze_fn, sym in [(analyze_xauusd, "XAUUSD"), (analyze_us100, "US100.cash")]:
            result = await analyze_fn(connection)
            if result:
                direction, price, tp, sl, ind, ind_name = result
                key = f"{direction}_{round(price, 1)}"
                if last_signal[sym] != key:
                    msg = format_message(sym, direction, price, tp, sl, ind, ind_name)
                    await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg, parse_mode=ParseMode.MARKDOWN)
                    last_signal[sym] = key
                    log.info(f"Signal {sym}: {direction} @ {price}")
            else:
                last_signal[sym] = None
    except Exception as e:
        log.error(f"Erreur scan: {e}")

    await asyncio.sleep(SCAN_INTERVAL)
```

if **name** == “**main**”:
asyncio.run(main())
