# XauBot Signal Bot - Railway / Twelve Data
# v20 : FVG 15M + OB 15M + LIMIT AUTO + FUTURES OFFSET + M5 REFINEMENT

import asyncio, logging, os, time, requests, pandas as pd
from datetime import datetime, timezone
from telegram import Bot
from telegram.request import HTTPXRequest

TELEGRAM_TOKEN   = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
TWELVE_API_KEY   = os.environ["TWELVE_API_KEY"]

SCAN_INTERVAL = 180
SIGNAL_COOLDOWN = 1800  # 30 min anti-spam

FUTURES_OFFSET = 0  # Mettre 0 si FTMO (spot XAU/USD)

XAUUSD_CONFIG = {
    "symbol": "XAU/USD", "label": "XAUUSD (MGC offset +{})".format(FUTURES_OFFSET),
    "ema_fast": 15, "ema_slow": 50,
    "adx_period": 14, "adx_min": 25,
    "atr_period": 14, "atr_sl_mult": 1.5,
    "fibo_lookback": 50,
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

def is_market_open():
    now_utc = datetime.now(timezone.utc)
    wd = now_utc.weekday(); h = now_utc.hour
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
            log.error("Twelve Data erreur " + symbol + ": " + str(data.get("message", ""))); return None
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
    return 100 - (100 / (1 + gain/loss))

def recommend_fibo(rsi_v, adx_v):
    if rsi_v > 70 and adx_v > 35: return "38.2"
    elif rsi_v >= 65 or adx_v >= 30: return "50.0"
    else: return "61.8"

def detect_fvg(df, lookback=30):
    n = len(df); end = n - 1
    for i in range(end - 1, max(2, end - lookback), -1):
        h0 = float(df["high"].iloc[i-2]); l0 = float(df["low"].iloc[i-2])
        hi = float(df["high"].iloc[i]);   li = float(df["low"].iloc[i])
        if li > h0: return ("BULL", round(h0,2), round(li,2))
        if hi < l0: return ("BEAR", round(hi,2), round(l0,2))
    return None

def detect_ob(df, lookback=30, atr_mult=1.5):
    n = len(df); atr_v = float(atr(df).iloc[-1]); end = n - 1
    for i in range(end-4, max(0, end-lookback), -1):
        o=float(df["open"].iloc[i]); c=float(df["close"].iloc[i])
        h=float(df["high"].iloc[i]); l=float(df["low"].iloc[i])
        fu_h = max(float(df["high"].iloc[j]) for j in range(i+1, min(i+4,n)))
        fu_l = min(float(df["low"].iloc[j])  for j in range(i+1, min(i+4,n)))
        if c < o and (fu_h-h) > atr_mult*atr_v: return ("BULL", round(l,2), round(h,2))
        if c > o and (l-fu_l) > atr_mult*atr_v: return ("BEAR", round(l,2), round(h,2))
    return None

def refine_entry_m5(symbol, direction, limit_entry):
    if limit_entry is None: return limit_entry
    lp = limit_entry["limit"]
    zone_low = lp - 15.0; zone_high = lp + 15.0
    target_type = "BULL" if direction == "BUY" else "BEAR"
    df5 = get_candles(symbol, interval="5min", outputsize=60)
    if df5 is None or len(df5) < 10:
        limit_entry["refined"] = False; limit_entry["m5_source"] = None; return limit_entry
    fvg5 = detect_fvg(df5)
    if fvg5 and fvg5[0] == target_type:
        fvg_low, fvg_high = fvg5[1], fvg5[2]
        if fvg_low <= zone_high + 5 and fvg_high >= zone_low - 5:
            entry = round(fvg_high if direction == "BUY" else fvg_low, 2)
            new_sl = round(fvg_low - 2.0, 2) if direction == "BUY" else round(fvg_high + 2.0, 2)
            limit_entry["limit"] = entry; limit_entry["sl"] = new_sl
            limit_entry["refined"] = True; limit_entry["m5_source"] = "FVG M5"
            return limit_entry
    ob5 = detect_ob(df5)
    if ob5 and ob5[0] == target_type:
        ob_low, ob_high = ob5[1], ob5[2]
        if ob_low <= zone_high + 5 and ob_high >= zone_low - 5:
            entry = round(ob_high if direction == "BUY" else ob_low, 2)
            new_sl = round(ob_low - 2.0, 2) if direction == "BUY" else round(ob_high + 2.0, 2)
            limit_entry["limit"] = entry; limit_entry["sl"] = new_sl
            limit_entry["refined"] = True; limit_entry["m5_source"] = "OB M5"
            return limit_entry
    limit_entry["refined"] = False; limit_entry["m5_source"] = None
    return limit_entry

def double_impulse(df):
    bull = (df["close"].iloc[-2]>df["open"].iloc[-2]) and (df["close"].iloc[-3]>df["open"].iloc[-3])
    bear = (df["close"].iloc[-2]<df["open"].iloc[-2]) and (df["close"].iloc[-3]<df["open"].iloc[-3])
    return bull, bear

def live_breakout(df, atr_v):
    c1=float(df["close"].iloc[-1]); o1=float(df["open"].iloc[-1])
    c2=float(df["close"].iloc[-2]); o2=float(df["open"].iloc[-2])
    b1=abs(c1-o1); b2=abs(c2-o2)
    if (c1>o1) and (b1>atr_v*2.0): return True,False,False
    if (c1<o1) and (b1>atr_v*2.0): return False,True,False
    if (c2>o2) and (b2>atr_v*2.0) and (c1>o1) and (b1>atr_v*0.3): return True,False,True
    if (c2<o2) and (b2>atr_v*2.0) and (c1<o1) and (b1>atr_v*0.3): return False,True,True
    return False,False,False

def detect_pattern(df):
    o=float(df["open"].iloc[-2]); h=float(df["high"].iloc[-2])
    l=float(df["low"].iloc[-2]);  c=float(df["close"].iloc[-2])
    o2=float(df["open"].iloc[-3]); c2=float(df["close"].iloc[-3])
    total=h-l
    if total==0: return None
    body=abs(c-o); uw=h-max(o,c); lw=min(o,c)-l
    bp=body/total; up=uw/total; lp=lw/total
    if bp<0.1: return "Doji"
    if bp>0.85: return "Marubozu "+("Bull" if c>o else "Bear")
    if lp>0.6 and bp<0.3: return "Pin Bar Bull"
    if up>0.6 and bp<0.3: return "Pin Bar Bear"
    if c>o and c2<o2 and c>o2 and o<c2: return "Engulfing Bull"
    if c<o and c2>o2 and c<o2 and o>c2: return "Engulfing Bear"
    return None

def fibo_range(df, lookback=50):
    r=df.tail(lookback)
    return round(float(r["high"].max()),2), round(float(r["low"].min()),2)

def fibonacci_levels(sh, sl):
    d=sh-sl
    if d==0: return None
    return {"38.2":round(sh-0.382*d,2),"50.0":round(sh-0.500*d,2),"61.8":round(sh-0.618*d,2)}

def nearest_fibo(price, fib):
    if fib is None: return "---"
    best,dist=None,float("inf")
    for k in ["38.2","50.0","61.8"]:
        d=abs(price-fib[k])
        if d<dist: dist,best=d,k
    return best

def get_htf_trend(symbol):
    df=get_candles(symbol,interval="30min",outputsize=60)
    if df is None or len(df)<55: return None
    df["ef"]=ema(df["close"],15); df["es"]=ema(df["close"],50)
    return "BULL" if float(df["ef"].iloc[-1])>float(df["es"].iloc[-1]) else "BEAR"

def calc_limit_entry(direction, price, fib, fvg, ob, atr_v):
    candidates=[]
    fib_scores={"38.2":10,"50.0":15,"61.8":25}
    fib_list=[("38.2",fib["38.2"]),("50.0",fib["50.0"]),("61.8",fib["61.8"])] if fib else []
    for lvl_name,lvl_price in fib_list:
        if direction=="BUY" and lvl_price>=price: continue
        if direction=="SELL" and lvl_price<=price: continue
        score=fib_scores.get(lvl_name,10); has_fvg=False; has_ob=False
        zone_bottom=lvl_price-3.0
        if fvg and fvg[0]==("BULL" if direction=="BUY" else "BEAR"):
            fvg_low,fvg_high=fvg[1],fvg[2]
            if fvg_low-8<=lvl_price<=fvg_high+8:
                score+=35; has_fvg=True; zone_bottom=min(zone_bottom,fvg_low-2.0)
        if ob and ob[0]==("BULL" if direction=="BUY" else "BEAR"):
            ob_low,ob_high=ob[1],ob[2]
            if ob_low-8<=lvl_price<=ob_high+8:
                score+=40; has_ob=True; zone_bottom=min(zone_bottom,ob_low-2.0)
        sl=round(zone_bottom-atr_v*0.3,2) if direction=="BUY" else round(lvl_price+atr_v*0.3+3.0,2)
        candidates.append({"fib":lvl_name,"limit":round(lvl_price,2),"sl":sl,"score":score,"has_fvg":has_fvg,"has_ob":has_ob})
    if not candidates:
        if direction=="BUY":
            if ob and ob[0]=="BULL":
                s=40+(35 if fvg and fvg[0]=="BULL" else 0)
                candidates.append({"fib":"OB","limit":round(ob[2],2),"sl":round(ob[1]-2.0,2),"score":s,"has_fvg":fvg is not None and fvg[0]=="BULL","has_ob":True})
            elif fvg and fvg[0]=="BULL":
                candidates.append({"fib":"FVG","limit":round(fvg[2],2),"sl":round(fvg[1]-2.0,2),"score":35,"has_fvg":True,"has_ob":False})
        else:
            if ob and ob[0]=="BEAR":
                s=40+(35 if fvg and fvg[0]=="BEAR" else 0)
                candidates.append({"fib":"OB","limit":round(ob[1],2),"sl":round(ob[2]+2.0,2),"score":s,"has_fvg":fvg is not None and fvg[0]=="BEAR","has_ob":True})
            elif fvg and fvg[0]=="BEAR":
                candidates.append({"fib":"FVG","limit":round(fvg[1],2),"sl":round(fvg[2]+2.0,2),"score":35,"has_fvg":True,"has_ob":False})
    if not candidates: return None
    best=sorted(candidates,key=lambda x:x["score"],reverse=True)[0]
    best["confidence_label"]="HIGH" if best["score"]>=75 else ("MEDIUM" if best["score"]>=55 else "LOW")
    best["refined"]=False; best["m5_source"]=None
    return best

def format_limit_message(label, direction, price, limit_entry, atr_v):
    if limit_entry is None: return None
    lp_spot=limit_entry["limit"]; sl_spot=limit_entry["sl"]
    conf=limit_entry["confidence_label"]; score=limit_entry["score"]; fib=limit_entry["fib"]
    refined=limit_entry.get("refined",False); m5_src=limit_entry.get("m5_source",None)
    off=FUTURES_OFFSET if direction=="BUY" else -FUTURES_OFFSET
    lp=round(lp_spot+off,2); sl=round(sl_spot+off,2)
    if direction=="BUY":
        tp1=round(lp+atr_v*0.8,2); tp2=round(lp+atr_v*1.5,2); tp3=round(lp+atr_v*2.5,2)
    else:
        tp1=round(lp-atr_v*0.8,2); tp2=round(lp-atr_v*1.5,2); tp3=round(lp-atr_v*2.5,2)
    conf_icon="🟢" if conf=="HIGH" else ("🟡" if conf=="MEDIUM" else "🔴")
    arrow="🟢" if direction=="BUY" else "🔴"
    fib_label=("Fibo "+str(fib)+"%") if fib not in ("OB","FVG") else fib
    precision_icon="🔬" if refined else "📐"
    precision_label=m5_src if refined else "Zone M15"
    msg ="📌 ORDRE LIMIT — "+label+"\n"
    msg+="━━━━━━━━━━━━━━━━━━\n"
    msg+=arrow+" "+direction+" LIMIT @ "+str(lp)+"\n"
    msg+="🛑 SL       : "+str(sl)+"\n"
    msg+="━━━━━━━━━━━━━━━━━━\n"
    msg+="🎯 TP1      : "+str(tp1)+"\n"
    msg+="🎯 TP2      : "+str(tp2)+"\n"
    msg+="🎯 TP3      : "+str(tp3)+"\n"
    msg+="━━━━━━━━━━━━━━━━━━\n"
    msg+="📐 Zone M15 : "+fib_label+"\n"
    msg+=precision_icon+" Entree     : "+precision_label+"\n"
    if limit_entry["has_fvg"]: msg+="✅ FVG M15 alignee\n"
    if limit_entry["has_ob"]:  msg+="✅ OB M15 aligne\n"
    msg+=conf_icon+" Confiance : "+conf+" ("+str(score)+"/100)\n"
    if FUTURES_OFFSET != 0:
        msg+="━━━━━━━━━━━━━━━━━━\n"
        msg+="📌 Prix ajustes futures (+"+str(FUTURES_OFFSET)+"$ offset)\n"
    msg+="⏳ Invalide si cloture sous SL ou en 20 bougies M15"
    return msg

def analyze_xauusd():
    cfg=XAUUSD_CONFIG
    htf=get_htf_trend(cfg["symbol"])
    if not htf: return None
    df=get_candles(cfg["symbol"])
    if df is None or len(df)<60: return None
    df["ef"]=ema(df["close"],cfg["ema_fast"]); df["es"]=ema(df["close"],cfg["ema_slow"])
    df["atr_v"]=atr(df,cfg["atr_period"])
    adx_s,pdi,mdi=adx(df,cfg["adx_period"])
    rsi_s=rsi(df["close"],14)
    price=round(float(df["close"].iloc[-1]),2)
    ef,es=float(df["ef"].iloc[-1]),float(df["es"].iloc[-1])
    adx_v=float(adx_s.iloc[-1]); pdi_v=float(pdi.iloc[-1]); mdi_v=float(mdi.iloc[-1])
    rsi_v=round(float(rsi_s.iloc[-1]),1)
    atr_v=float(df["atr_v"].iloc[-1])
    bull_i,bear_i=double_impulse(df)
    bull_live,bear_live,confirmed=live_breakout(df,atr_v)
    sh,sl_s=fibo_range(df,cfg["fibo_lookback"])
    fib=fibonacci_levels(sh,sl_s)
    fib_lvl=nearest_fibo(price,fib)
    pattern=detect_pattern(df)
    df15=get_candles(cfg["symbol"],interval="15min",outputsize=60)
    fvg=detect_fvg(df15) if df15 is not None and len(df15)>=10 else None
    ob=detect_ob(df15)   if df15 is not None and len(df15)>=10 else None
    sd=round(atr_v*cfg["atr_sl_mult"],2)
    tp1_d=round(atr_v*0.8,2); tp2_d=round(atr_v*1.5,2); tp3_d=round(atr_v*2.5,2)
    if pdi_v>mdi_v and adx_v>cfg["adx_min"] and bull_live:
        st="BREAKOUT_CONF" if confirmed else "BREAKOUT"
        return ("BUY",price,round(price-sd,2),round(price+tp1_d,2),round(price+tp2_d,2),round(price+tp3_d,2),round(adx_v,1),htf,fib,fib_lvl,st,pattern,rsi_v,fvg,ob,atr_v)
    if mdi_v>pdi_v and adx_v>cfg["adx_min"] and bear_live:
        st="BREAKOUT_CONF" if confirmed else "BREAKOUT"
        return ("SELL",price,round(price+sd,2),round(price-tp1_d,2),round(price-tp2_d,2),round(price-tp3_d,2),round(adx_v,1),htf,fib,fib_lvl,st,pattern,rsi_v,fvg,ob,atr_v)
    if ef>es and pdi_v>mdi_v and adx_v>cfg["adx_min"] and bull_i and htf=="BULL":
        return ("BUY",price,round(price-sd,2),round(price+tp1_d,2),round(price+tp2_d,2),round(price+tp3_d,2),round(adx_v,1),htf,fib,fib_lvl,"SIGNAL",pattern,rsi_v,fvg,ob,atr_v)
    if ef<es and mdi_v>pdi_v and adx_v>cfg["adx_min"] and bear_i and htf=="BEAR":
        return ("SELL",price,round(price+sd,2),round(price-tp1_d,2),round(price-tp2_d,2),round(price-tp3_d,2),round(adx_v,1),htf,fib,fib_lvl,"SIGNAL",pattern,rsi_v,fvg,ob,atr_v)
    return None

def format_message(label,direction,price,sl,tp1,tp2,tp3,val,htf,fib,fib_lvl,signal_type="SIGNAL",pattern=None,rsi_v=None,fvg=None,ob=None):
    now=datetime.utcnow().strftime("%H:%M UTC")
    arrow="🟢" if direction=="BUY" else "🔴"
    icon="✅" if (direction=="BUY" and htf=="BULL") or (direction=="SELL" and htf=="BEAR") else "⚠️"
    sl_d=round(abs(price-sl),2)
    if signal_type in ("BREAKOUT","BREAKOUT_CONF"):
        msg="⚡ BREAKOUT "+direction+" - "+label+"\n"+"━━━━━━━━━━━━━━━━━━\n"+"🔥 Rupture en cours — preparer le retracement\n"+"━━━━━━━━━━━━━━━━━━\n"
    else:
        msg=arrow+" "+direction+" SIGNAL - "+label+"\n"+"━━━━━━━━━━━━━━━━━━\n"
    msg+="🕐 Heure  : "+now+"\n📍 Entry  : "+str(price)+"\n🛑 SL     : "+str(sl)+"  (-"+str(sl_d)+")\n"
    msg+="━━━━━━━━━━━━━━━━━━\n"
    msg+="🎯 TP1    : "+str(tp1)+"  (securiser 50%)\n🎯 TP2    : "+str(tp2)+"  (RR 1:1)\n🎯 TP3    : "+str(tp3)+"  (RR 1:1.7)\n"
    msg+="━━━━━━━━━━━━━━━━━━\n📊 ADX    : "+str(val)+"\n"
    if rsi_v is not None: msg+="📉 RSI    : "+str(rsi_v)+"\n"
    if fvg:
        fvg_icon="✅" if (fvg[0]=="BULL" and direction=="BUY") or (fvg[0]=="BEAR" and direction=="SELL") else "⚠️"
        prox_fvg=" 📍PRIX DEDANS" if fvg[1]<=price<=fvg[2] else (" 🔜APPROCHE" if abs(price-(fvg[1]+fvg[2])/2)<20 else "")
        msg+="📐 FVG 15M : "+fvg[0]+" "+fvg_icon+" ["+str(fvg[1])+"-"+str(fvg[2])+"]"+prox_fvg+"\n"
    else: msg+="📐 FVG 15M : -\n"
    if ob:
        ob_icon="✅" if (ob[0]=="BULL" and direction=="BUY") or (ob[0]=="BEAR" and direction=="SELL") else "⚠️"
        prox_ob=" 📍PRIX DEDANS" if ob[1]<=price<=ob[2] else (" 🔜APPROCHE" if abs(price-(ob[1]+ob[2])/2)<20 else "")
        msg+="📦 OB 15M  : "+ob[0]+" "+ob_icon+" ["+str(ob[1])+"-"+str(ob[2])+"]"+prox_ob+"\n"
    else: msg+="📦 OB 15M  : -\n"
    if fvg and ob and fvg[0]==ob[0] and ((fvg[0]=="BULL" and direction=="BUY") or (fvg[0]=="BEAR" and direction=="SELL")):
        msg+="🔥 Confluence FVG+OB — zone tres forte !\n"
    msg+="📈 M30    : "+htf+" "+icon+"\n"
    if pattern: msg+="🕯 Pattern : "+pattern+"\n"
    msg+="━━━━━━━━━━━━━━━━━━\n"
    if fib:
        rec=recommend_fibo(rsi_v,val) if rsi_v is not None else "50.0"
        for lvl in ["38.2","50.0","61.8"]:
            star=" ⭐ RECOMMANDE" if lvl==rec else ""
            msg+="📐 Fibo "+lvl+"% : "+str(fib[lvl])+star+"\n"
    msg+="━━━━━━━━━━━━━━━━━━\n"
    if signal_type=="BREAKOUT_CONF": msg+="✅ Breakout confirme — attends le retracement Fibo"
    elif signal_type=="BREAKOUT": msg+="⚠️ Bougie en cours — attends la cloture et le retracement"
    else: msg+="📌 Attends le retracement — ordre limit sur M5 ⭐"
    return msg

last_signal={"XAUUSD":{"direction":None,"type":None,"ts":0}}

async def main():
    bot=Bot(token=TELEGRAM_TOKEN, request=HTTPXRequest(read_timeout=30,connect_timeout=30,write_timeout=30))
    for attempt in range(5):
        try:
            await bot.send_message(chat_id=TELEGRAM_CHAT_ID,
                text="XauBot Signal v20 demarre\nScan 3min | Lun-Jeu 8h-19h UTC | Ven 8h-17h UTC | ADX 25 | RSI + Fibo | FVG 15M | OB 15M | M30 | Cooldown 30min | LIMIT AUTO | M5 REFINEMENT | FUTURES OFFSET +57.8$")
            break
        except Exception as e:
            log.error(f"Startup msg attempt {attempt+1}: {e}")
            await asyncio.sleep(10)
    log.info("Bot demarre v20")
    while True:
        try:
            if not is_market_open():
                log.info("Marche ferme"); await asyncio.sleep(SCAN_INTERVAL); continue
            xau=analyze_xauusd()
            if xau:
                d,p,sl,tp1,tp2,tp3,v,htf,fib,fl,st,pat,rsiv,fvg,ob,atr_v=xau
                prev=last_signal["XAUUSD"]
                elapsed=time.time()-prev["ts"]
                same=(prev["direction"]==d)  # cooldown sur direction uniquement
                if not same or elapsed>SIGNAL_COOLDOWN:
                    await bot.send_message(chat_id=TELEGRAM_CHAT_ID,
                        text=format_message("XAUUSD",d,p,sl,tp1,tp2,tp3,v,htf,fib,fl,st,pat,rsiv,fvg,ob))
                    last_signal["XAUUSD"]={"direction":d,"type":st,"ts":time.time()}
                    await asyncio.sleep(2)
                    limit_data=calc_limit_entry(d,p,fib,fvg,ob,atr_v)
                    limit_data=refine_entry_m5(XAUUSD_CONFIG["symbol"],d,limit_data)
                    limit_msg=format_limit_message("XAUUSD",d,p,limit_data,atr_v)
                    if limit_msg:
                        await bot.send_message(chat_id=TELEGRAM_CHAT_ID,text=limit_msg)
        except Exception as e:
            log.error("Erreur: "+str(e))
        await asyncio.sleep(SCAN_INTERVAL)

if __name__=="__main__":
    asyncio.run(main())
