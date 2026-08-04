"""
Gold Real-time Scanner (scanner.py)
สคริปต์สแกนราคาทองคำ Real-time M5 SMC (Wave 3 + FVG)
รองรับ Auto-Config, ML Model Filtering และการส่งสัญญาณเข้า Telegram
"""

import os
import math
import json
import requests
import joblib
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# ==================== CONFIGURATION ====================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", os.getenv("TELEGRAM_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN_HERE"))
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "YOUR_TELEGRAM_CHAT_ID_HERE")
MODEL_FILE_PATH    = os.getenv("MODEL_FILE_PATH", "gold_ml_filter.pkl")
CONFIG_FILE_PATH   = os.getenv("CONFIG_FILE_PATH", "best_config.json")

SYMBOL              = "GC=F"                                        # Spot Gold / Gold Futures บน Yahoo
ACCOUNT_EQUITY      = float(os.getenv("ACCOUNT_EQUITY", "10000.0")) # ขนาดพอร์ต ($)
BASE_RISK_PCT       = float(os.getenv("RISK_PCT", "0.01"))          # ความเสี่ยงพื้นฐาน 1% (0.01)
DYNAMIC_LOT_SCALING = True                                         # เปิดใช้การปรับ Lot ตามความมั่นใจโมเดล

def load_auto_config():
    """ โหลดพารามิเตอร์ที่ผ่านการ Auto-Optimization จาก best_config.json """
    default_config = {
        "PROBA_THRESHOLD": 0.60,
        "RR_RATIO": 3.0,
        "GOLD_SL_BUFFER": 0.80
    }
    
    if os.path.exists(CONFIG_FILE_PATH):
        try:
            with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
                config = json.load(f)
                print(f"⚙️ Loaded Config from '{CONFIG_FILE_PATH}':")
                print(f"   • Proba Threshold : {config.get('PROBA_THRESHOLD', default_config['PROBA_THRESHOLD'])}")
                print(f"   • RR Ratio        : 1:{config.get('RR_RATIO', default_config['RR_RATIO'])}")
                print(f"   • SL Buffer       : ${config.get('GOLD_SL_BUFFER', default_config['GOLD_SL_BUFFER'])}")
                return config
        except Exception as e:
            print(f"⚠️ อ่านไฟล์คอนฟิกไม่สำเร็จ ({e}) -> ใช้งานค่า Default")
            return default_config
    else:
        print(f"⚠️ ไม่พบไฟล์ {CONFIG_FILE_PATH} -> ใช้งานค่า Default")
        return default_config

def send_telegram_alert(message: str):
    """ ส่งข้อความการแจ้งเตือนเข้า Telegram """
    if TELEGRAM_BOT_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN_HERE" or TELEGRAM_CHAT_ID == "YOUR_TELEGRAM_CHAT_ID_HERE":
        print("ℹ️ ไม่ได้ระบุ TELEGRAM_BOT_TOKEN หรือ TELEGRAM_CHAT_ID (แสดงผลเฉพาะบน Console)")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        response = requests.post(url, json=payload, timeout=10)
        res_data = response.json()
        if res_data.get("ok"):
            print("🔔 ส่งการแจ้งเตือนไปยัง Telegram เรียบร้อยแล้ว")
        else:
            print(f"❌ ส่ง Telegram ไม่สำเร็จ: {res_data.get('description')}")
    except Exception as e:
        print(f"❌ ไม่สามารถเชื่อมต่อ Telegram ได้: {e}")

def calculate_lot_size(equity: float, risk_dist: float, win_prob: float) -> tuple[float, float]:
    """ คำนวณ Lot Size โดยอิงจากระยะ SL และความมั่นใจของโมเดล """
    if DYNAMIC_LOT_SCALING and win_prob >= 0.65:
        applied_risk_pct = BASE_RISK_PCT * 1.5  # มั่นใจสูง (>=65%): เพิ่มความเสี่ยงเป็น 1.5%
    else:
        applied_risk_pct = BASE_RISK_PCT        # มั่นใจปกติ: ใช้ความเสี่ยงพื้นฐาน 1.0%

    risk_amount = equity * applied_risk_pct
    calculated_lot = risk_amount / (risk_dist * 100.0)
    
    final_lot = math.floor(calculated_lot * 100.0) / 100.0
    final_lot = max(0.01, final_lot)
    
    return final_lot, applied_risk_pct * 100.0

def run_scanner():
    # โหลด Config อัตโนมัติจาก best_config.json
    config = load_auto_config()
    proba_threshold = float(config.get("PROBA_THRESHOLD", 0.60))
    rr_ratio        = float(config.get("RR_RATIO", 3.0))
    sl_buffer       = float(config.get("GOLD_SL_BUFFER", 0.80))

    current_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n🔍 [{current_time_str}] สแกนกราฟ M5 (Threshold: {proba_threshold*100:.0f}%)...")

    if not os.path.exists(MODEL_FILE_PATH):
        print(f"❌ ไม่พบไฟล์โมเดลที่ Path: {MODEL_FILE_PATH}")
        return

    try:
        model = joblib.load(MODEL_FILE_PATH)
    except Exception as e:
        print(f"❌ โหลดโมเดลไม่สำเร็จ: {e}")
        return

    try:
        df_5m = yf.download(SYMBOL, period="5d", interval="5m", progress=False)
        if isinstance(df_5m.columns, pd.MultiIndex):
            df_5m.columns = df_5m.columns.get_level_values(0)

        df_5m = df_5m.dropna()
        if len(df_5m) < 50:
            print("❌ ข้อมูลไม่เพียงพอสำหรับประมวลผล")
            return

        # Multi-timeframe Resampling (H1 และ M15)
        df_15m = df_5m.resample('15min').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last'}).dropna()
        df_1h  = df_5m.resample('1H').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last'}).dropna()

        # Trend Indicators
        df_1h['EMA_50'] = df_1h['Close'].ewm(span=50, adjust=False).mean()
        df_1h['H1_Trend'] = np.where(df_1h['Close'] > df_1h['EMA_50'], 1, -1)

        df_15m['EMA_20'] = df_15m['Close'].ewm(span=20, adjust=False).mean()
        df_15m['M15_Trend'] = np.where(df_15m['Close'] > df_15m['EMA_20'], 1, -1)

        df_5m['H1_Trend'] = df_1h['H1_Trend'].reindex(df_5m.index, method='ffill')
        df_5m['M15_Trend'] = df_15m['M15_Trend'].reindex(df_5m.index, method='ffill')

        # ATR & Fair Value Gap (FVG)
        high_low = df_5m['High'] - df_5m['Low']
        high_cp  = np.abs(df_5m['High'] - df_5m['Close'].shift(1))
        low_cp   = np.abs(df_5m['Low'] - df_5m['Close'].shift(1))
        df_5m['ATR'] = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1).rolling(14).mean()

        df_5m['Bullish_FVG'] = df_5m['Low'] > df_5m['High'].shift(2)
        df_5m['Bearish_FVG'] = df_5m['High'] < df_5m['Low'].shift(2)

        latest_bar  = df_5m.iloc[-2]
        prev_bar    = df_5m.iloc[-3]
        latest_time = df_5m.index[-2]

        is_long  = (latest_bar['H1_Trend'] == 1) and (latest_bar['M15_Trend'] == 1) and latest_bar['Bullish_FVG']
        is_short = (latest_bar['H1_Trend'] == -1) and (latest_bar['M15_Trend'] == -1) and latest_bar['Bearish_FVG']

        if not (is_long or is_short):
            print("ℹ️ ไม่พบ Setup SMC ในแท่งปัจจุบัน")
            return

        entry_price = float(latest_bar['Close'])
        if is_long:
            sl_price    = float(min(latest_bar['Low'], prev_bar['Low'])) - sl_buffer
            risk_dist   = entry_price - sl_price
            tp_price    = entry_price + (risk_dist * rr_ratio)
            fvg_size    = float(latest_bar['Low'] - df_5m.iloc[-4]['High'])
            signal_type = "BUY 🟢"
        else:
            sl_price    = float(max(latest_bar['High'], prev_bar['High'])) + sl_buffer
            risk_dist   = sl_price - entry_price
            tp_price    = entry_price - (risk_dist * rr_ratio)
            fvg_size    = float(df_5m.iloc[-4]['Low'] - latest_bar['High'])
            signal_type = "SELL 🔴"

        if risk_dist <= 0:
            print("⚠️ ระยะ Risk Distance ไม่ถูกต้อง (<= 0) ข้ามการประมวลผล")
            return

        features = pd.DataFrame([{
            'FVG_Size': fvg_size,
            'ATR': float(latest_bar['ATR']),
            'Hour': int(latest_time.hour),
            'Minute': int(latest_time.minute),
            'Risk_Distance': risk_dist
        }])

        win_prob = float(model.predict_proba(features)[0][1])
        print(f"💡 พบ Setup {signal_type} | ML Win Prob: {win_prob*100:.1f}%")

        # กรองเฉพาะสัญญาณที่ผ่าน Threshold
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
                f"⏰ *Time:* `{latest_time.strftime('%Y-%m-%d %H:%M')}`\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"💡 _ยกระดับด้วย H1-M15-M5 SMC + Dynamic ML Risk_"
            )

            print(alert_msg)
            send_telegram_alert(alert_msg)
        else:
            print(f"⛔ สัญญาณถูกปฏิเสธเนื่องจากความมั่นใจของ ML ({win_prob*100:.1f}%) ต่ำกว่าเกณฑ์ที่ตั้งไว้ ({proba_threshold*100:.1f}%)")

    except Exception as e:
        print(f"⚠️ เกิดข้อผิดพลาดขณะสแกนตลาด: {e}")

if __name__ == "__main__":
    run_scanner()
            
