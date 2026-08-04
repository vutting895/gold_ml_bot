"""
Gold Real-time Scanner (scanner.py)
สคริปต์สแกนราคาทองคำ Real-time M5 SMC (Wave 3 + FVG)
พร้อมระบบ Deduplication (กันสัญญาณซ้ำ), News Filter และ ML Model Filtering
"""

import os
import math
import json
import requests
import joblib
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, timezone

# ==================== CONFIGURATION ====================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", os.getenv("TELEGRAM_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN_HERE"))
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "YOUR_TELEGRAM_CHAT_ID_HERE")
MODEL_FILE_PATH    = os.getenv("MODEL_FILE_PATH", "gold_ml_filter.pkl")
CONFIG_FILE_PATH   = os.getenv("CONFIG_FILE_PATH", "best_config.json")
LAST_SIGNAL_PATH   = "last_signal.json"

SYMBOL              = "GC=F"
ACCOUNT_EQUITY      = float(os.getenv("ACCOUNT_EQUITY", "10000.0"))
BASE_RISK_PCT       = float(os.getenv("RISK_PCT", "0.01"))
DYNAMIC_LOT_SCALING = True

# ==================== PRIORITY 2: NEWS FILTER ====================
def is_high_impact_news_near(buffer_minutes: int = 30) -> tuple[bool, str]:
    """ ดึงข้อมูลปฏิทินข่าว High Impact (USD/XAU) จาก Forex Factory และตรวจสอบช่วงเวลาใกล้ข่าว """
    url = "https://nfp.ourforecast.com/api/v1/forexfactory"  # API ปฏิทินข่าวสาธารณะ
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            events = response.json()
            now_utc = datetime.now(timezone.utc)

            for event in events:
                if event.get("currency") in ["USD", "XAU"] and event.get("impact") == "High":
                    event_time_str = event.get("date")  # ISO Format
                    if event_time_str:
                        event_dt = datetime.fromisoformat(event_time_str.replace("Z", "+00:00"))
                        time_diff = abs((event_dt - now_utc).total_seconds() / 60.0)
                        
                        if time_diff <= buffer_minutes:
                            title = event.get('title', 'High Impact News')
                            return True, f"{title} ({event.get('currency')})"
    except Exception as e:
        print(f"⚠️ ไม่สามารถเชื่อมต่อ API ข่าวได้ ({e}) -> ดำเนินการสแกนต่อตามปกติ")
    return False, ""

# ==================== PRIORITY 1: SIGNAL DEDUPLICATION ====================
def is_duplicate_signal(signal_time_str: str, signal_type: str) -> bool:
    """ ตรวจสอบว่าสัญญาณบนแท่งเวลาและทิศทางนี้เคยส่งไปแล้วหรือยัง """
    if os.path.exists(LAST_SIGNAL_PATH):
        try:
            with open(LAST_SIGNAL_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data.get("time") == signal_time_str and data.get("type") == signal_type:
                    return True
        except Exception:
            pass
    return False

def save_last_signal(signal_time_str: str, signal_type: str):
    """ บันทึกสัญญาณล่าสุดลงไฟล์เพื่อกันส่งซ้ำ """
    try:
        with open(LAST_SIGNAL_PATH, "w", encoding="utf-8") as f:
            json.dump({"time": signal_time_str, "type": signal_type}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ ไม่สามารถบันทึก last_signal.json ได้: {e}")

# ==================== HELPER FUNCTIONS ====================
def load_auto_config():
    default_config = {"PROBA_THRESHOLD": 0.60, "RR_RATIO": 3.0, "GOLD_SL_BUFFER": 0.80}
    if os.path.exists(CONFIG_FILE_PATH):
        try:
            with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default_config
    return default_config

def send_telegram_alert(message: str):
    if TELEGRAM_BOT_TOKEN in ["YOUR_TELEGRAM_BOT_TOKEN_HERE", ""] or TELEGRAM_CHAT_ID in ["YOUR_TELEGRAM_CHAT_ID_HERE", ""]:
        print("ℹ️ ไม่ได้ระบุ TELEGRAM_BOT_TOKEN หรือ TELEGRAM_CHAT_ID")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        res = requests.post(url, json=payload, timeout=10).json()
        if res.get("ok"):
            print("🔔 ส่งการแจ้งเตือนไปยัง Telegram เรียบร้อยแล้ว")
        else:
            print(f"❌ ส่ง Telegram ไม่สำเร็จ: {res.get('description')}")
    except Exception as e:
        print(f"❌ ไม่สามารถเชื่อมต่อ Telegram ได้: {e}")

def calculate_lot_size(equity: float, risk_dist: float, win_prob: float) -> tuple[float, float]:
    applied_risk_pct = BASE_RISK_PCT * 1.5 if (DYNAMIC_LOT_SCALING and win_prob >= 0.65) else BASE_RISK_PCT
    risk_amount = equity * applied_risk_pct
    calculated_lot = risk_amount / (risk_dist * 100.0)
    final_lot = max(0.01, math.floor(calculated_lot * 100.0) / 100.0)
    return final_lot, applied_risk_pct * 100.0

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# ==================== MAIN SCANNER ====================
def run_scanner():
    config = load_auto_config()
    proba_threshold = float(config.get("PROBA_THRESHOLD", 0.60))
    rr_ratio        = float(config.get("RR_RATIO", 3.0))
    sl_buffer       = float(config.get("GOLD_SL_BUFFER", 0.80))

    current_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n🔍 [{current_time_str}] สแกนกราฟ M5 (Threshold: {proba_threshold*100:.0f}%)...")

    # Priority 2: ตรวจสอบปฏิทินข่าวใหญ่
    has_news, news_title = is_high_impact_news_near(buffer_minutes=30)
    if has_news:
        print(f"🛑 [NEWS FILTER] งดสแกนสัญญาณเนื่องจากใกล้ช่วงข่าวใหญ่: {news_title}")
        return

    if not os.path.exists(MODEL_FILE_PATH):
        print(f"❌ ไม่พบไฟล์โมเดลที่ Path: {MODEL_FILE_PATH}")
        return

    try:
        model = joblib.load(MODEL_FILE_PATH)
        df_5m = yf.download(SYMBOL, period="5d", interval="5m", progress=False)
        if isinstance(df_5m.columns, pd.MultiIndex):
            df_5m.columns = df_5m.columns.get_level_values(0)

        df_5m = df_5m.dropna()
        if len(df_5m) < 50:
            print("❌ ข้อมูลไม่เพียงพอสำหรับประมวลผล")
            return

        # Technical Indicators
        df_15m = df_5m.resample('15min').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last'}).dropna()
        df_1h  = df_5m.resample('1H').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last'}).dropna()

        df_1h['EMA_50'] = df_1h['Close'].ewm(span=50, adjust=False).mean()
        df_1h['H1_Trend'] = np.where(df_1h['Close'] > df_1h['EMA_50'], 1, -1)

        df_15m['EMA_20'] = df_15m['Close'].ewm(span=20, adjust=False).mean()
        df_15m['M15_Trend'] = np.where(df_15m['Close'] > df_15m['EMA_20'], 1, -1)

        df_5m['H1_Trend'] = df_1h['H1_Trend'].reindex(df_5m.index, method='ffill')
        df_5m['M15_Trend'] = df_15m['M15_Trend'].reindex(df_5m.index, method='ffill')

        # ATR & RSI (Features)
        high_low = df_5m['High'] - df_5m['Low']
        high_cp  = np.abs(df_5m['High'] - df_5m['Close'].shift(1))
        low_cp   = np.abs(df_5m['Low'] - df_5m['Close'].shift(1))
        df_5m['ATR'] = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1).rolling(14).mean()
        df_5m['RSI'] = calculate_rsi(df_5m['Close'], 14)

        df_5m['Bullish_FVG'] = df_5m['Low'] > df_5m['High'].shift(2)
        df_5m['Bearish_FVG'] = df_5m['High'] < df_5m['Low'].shift(2)

        latest_bar  = df_5m.iloc[-2]
        prev_bar    = df_5m.iloc[-3]
        latest_time = df_5m.index[-2]
        latest_time_str = latest_time.strftime('%Y-%m-%d %H:%M')

        is_long  = (latest_bar['H1_Trend'] == 1) and (latest_bar['M15_Trend'] == 1) and latest_bar['Bullish_FVG']
        is_short = (latest_bar['H1_Trend'] == -1) and (latest_bar['M15_Trend'] == -1) and latest_bar['Bearish_FVG']

        if not (is_long or is_short):
            print("ℹ️ ไม่พบ Setup SMC ในแท่งปัจจุบัน")
            return

        signal_type = "BUY 🟢" if is_long else "SELL 🔴"

        # Priority 1: ตรวจสอบการส่งซ้ำ
        if is_duplicate_signal(latest_time_str, signal_type):
            print(f"ℹ️ สัญญาณ {signal_type} ของแท่ง {latest_time_str} เคยถูกส่งไปแล้ว (Skip Duplicate)")
            return

        entry_price = float(latest_bar['Close'])
        if is_long:
            sl_price  = float(min(latest_bar['Low'], prev_bar['Low'])) - sl_buffer
            risk_dist = entry_price - sl_price
            tp_price  = entry_price + (risk_dist * rr_ratio)
            fvg_size  = float(latest_bar['Low'] - df_5m.iloc[-4]['High'])
        else:
            sl_price  = float(max(latest_bar['High'], prev_bar['High'])) + sl_buffer
            risk_dist = sl_price - entry_price
            tp_price  = entry_price - (risk_dist * rr_ratio)
            fvg_size  = float(df_5m.iloc[-4]['Low'] - latest_bar['High'])

        if risk_dist <= 0:
            return

        # Feature Set ที่ตรงกับ train_gold_model.py
        features = pd.DataFrame([{
            'FVG_Size': fvg_size,
            'ATR': float(latest_bar['ATR']),
            'RSI': float(latest_bar['RSI']),
            'Hour': int(latest_time.hour),
            'DayOfWeek': int(latest_time.dayofweek),
            'Risk_Distance': risk_dist
        }])

        win_prob = float(model.predict_proba(features)[0][1])
        print(f"💡 พบ Setup {signal_type} | ML Win Prob: {win_prob*100:.1f}%")

        if win_prob >= proba_threshold:
            lot_size, risk_pct_used = calculate_lot_size(ACCOUNT_EQUITY, risk_dist, win_prob)

            alert_msg = (
                f"🚨 *SIGNAL ALERT: XAU/USD ({signal_type})*\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"🤖 *ML Win Probability:* `{win_prob*100:.1f}%` (Threshold ≥ {proba_threshold*100:.0f}%)\n"
                f"📍 *Entry Price:* `${entry_price:,.2f}`\n"
                f"🛑 *Stop Loss (SL):* `${sl_price:,.2f}` (`${risk_dist:.2f}` Risk)\n"
                f"🎯 *Take Profit (TP):* `${tp_price:,.2f}` (R:R 1:{rr_ratio})\n"
                f"⚖️ *Recommended Lot:* `{lot_size}` Lot (Risk {risk_pct_used:.1f}%)\n"
                f"⏰ *Time:* `{latest_time_str}`\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"💡 _ยกระดับด้วย H1-M15-M5 SMC + Deduplication & News Filter_"
            )

            print(alert_msg)
            send_telegram_alert(alert_msg)
            save_last_signal(latest_time_str, signal_type)  # บันทึกกันส่งซ้ำ
        else:
            print(f"⛔ สัญญาณถูกปฏิเสธเนื่องจากความมั่นใจ ML ({win_prob*100:.1f}%) ต่ำกว่าเกณฑ์")

    except Exception as e:
        print(f"⚠️ เกิดข้อผิดพลาดขณะสแกนตลาด: {e}")

if __name__ == "__main__":
    run_scanner()
    
