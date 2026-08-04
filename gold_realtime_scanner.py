import os
import math
import requests
import joblib
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, time as dtime

# ==================== 1. TELEGRAM & MODEL CONFIGURATION ====================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN_HERE")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "YOUR_TELEGRAM_CHAT_ID_HERE")
MODEL_FILE_PATH = os.getenv("MODEL_FILE_PATH", "gold_ml_filter.pkl")

SYMBOL = "GC=F"              # Gold Futures / Spot Gold (XAUUSD)
RISK_PCT = 0.01              # Risk 1% ต่อไม้
ACCOUNT_EQUITY = 10000.0     # เงินทุนปัจจุบันในพอร์ต ($)
RR_RATIO = 3.0               # Risk-to-Reward Ratio (1:3.0)
GOLD_SL_BUFFER = 0.80        # SL Buffer เผื่อไส้เทียน ($0.80)
PROBABILITY_THRESHOLD = 0.55 # กรองเฉพาะความมั่นใจโมเดล >= 55%

def send_telegram_alert(message: str):
    """ ส่งข้อความการแจ้งเตือนเข้า Telegram """
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            print("📩 ส่งสัญญาณเข้า Telegram เรียบร้อยแล้ว!")
        else:
            print(f"⚠️ Telegram Error: {response.text}")
    except Exception as e:
        print(f"❌ ไม่สามารถเชื่อมต่อ Telegram ได้: {e}")

def scan_gold_market():
    current_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"🔍 [{current_time_str}] กำลังสแกนกราฟราคาทองคำ M5...")

    if not os.path.exists(MODEL_FILE_PATH):
        print(f"❌ ไม่พบไฟล์โมเดลที่ Path: {MODEL_FILE_PATH}")
        return

    try:
        model = joblib.load(MODEL_FILE_PATH)
        print("🧠 โหลดไฟล์โมเดลสำเร็จ!")
    except Exception as e:
        print(f"❌ ไม่สามารถโหลดโมเดลได้: {e}")
        return

    try:
        # ดึงข้อมูล M5 ย้อนหลัง 5 วันล่าสุดเพื่อคำนวณ Indicator
        df_5m = yf.download(SYMBOL, period="5d", interval="5m", progress=False)

        if isinstance(df_5m.columns, pd.MultiIndex):
            df_5m.columns = df_5m.columns.get_level_values(0)

        df_5m = df_5m.dropna()
        if len(df_5m) < 50:
            print("⚠️ ข้อมูลไม่เพียงพอสำหรับการวิเคราะห์")
            return

        # Resample เป็น M15 และ H1
        df_15m = df_5m.resample('15min').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last'}).dropna()
        df_1h  = df_5m.resample('1H').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last'}).dropna()

        # คำนวณ Indicators
        df_1h['EMA_50'] = df_1h['Close'].ewm(span=50, adjust=False).mean()
        df_1h['H1_Trend'] = np.where(df_1h['Close'] > df_1h['EMA_50'], 1, -1)

        df_15m['EMA_20'] = df_15m['Close'].ewm(span=20, adjust=False).mean()
        df_15m['M15_Trend'] = np.where(df_15m['Close'] > df_15m['EMA_20'], 1, -1)

        # Map ทรวดทรงเทรนด์กลับลงมา M5
        df_5m['H1_Trend'] = df_1h['H1_Trend'].reindex(df_5m.index, method='ffill')
        df_5m['M15_Trend'] = df_15m['M15_Trend'].reindex(df_5m.index, method='ffill')

        # คำนวณ ATR 14
        high_low = df_5m['High'] - df_5m['Low']
        high_cp = np.abs(df_5m['High'] - df_5m['Close'].shift(1))
        low_cp = np.abs(df_5m['Low'] - df_5m['Close'].shift(1))
        df_5m['ATR'] = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1).rolling(14).mean()

        # คำนวณ Fair Value Gap (FVG)
        df_5m['Bullish_FVG'] = df_5m['Low'] > df_5m['High'].shift(2)
        df_5m['Bearish_FVG'] = df_5m['High'] < df_5m['Low'].shift(2)

        # ตรวจสอบแท่งล่าสุดที่เพิ่งปิด (Index -2)
        latest_bar = df_5m.iloc[-2]
        prev_bar   = df_5m.iloc[-3]
        latest_time = df_5m.index[-2]

        # เช็คเงื่อนไข SMC Setup
        is_long = (latest_bar['H1_Trend'] == 1) and (latest_bar['M15_Trend'] == 1) and latest_bar['Bullish_FVG']
        is_short = (latest_bar['H1_Trend'] == -1) and (latest_bar['M15_Trend'] == -1) and latest_bar['Bearish_FVG']

        if not (is_long or is_short):
            print("ℹ️ ไม่พบ Setup ตามเงื่อนไขในแท่งปัจจุบัน")
            return

        # คำนวณ Features
        entry_price = float(latest_bar['Close'])
        if is_long:
            sl_price = float(min(latest_bar['Low'], prev_bar['Low'])) - GOLD_SL_BUFFER
            risk_dist = entry_price - sl_price
            tp_price = entry_price + (risk_dist * RR_RATIO)
            fvg_size = float(latest_bar['Low'] - df_5m.iloc[-4]['High'])
            signal_type = "BUY 🟢"
        else:
            sl_price = float(max(latest_bar['High'], prev_bar['High'])) + GOLD_SL_BUFFER
            risk_dist = sl_price - entry_price
            tp_price = entry_price - (risk_dist * RR_RATIO)
            fvg_size = float(df_5m.iloc[-4]['Low'] - latest_bar['High'])
            signal_type = "SELL 🔴"

        if risk_dist <= 0:
            return

        # จัดเตรียม Feature Vector สำหรับส่งให้โมเดลประเมิน
        features = pd.DataFrame([{
            'FVG_Size': fvg_size,
            'ATR': float(latest_bar['ATR']),
            'Hour': int(latest_time.hour),
            'Minute': int(latest_time.minute),
            'Risk_Distance': risk_dist
        }])

        # คำนวณค่าความน่าจะเป็นจาก Machine Learning
        win_prob = float(model.predict_proba(features)[0][1])
        print(f"💡 พบ Setup {signal_type} | ความมั่นใจของโมเดล: {win_prob*100:.1f}%")

        # กรองเฉพาะสัญญาณที่มีความมั่นใจ >= 55%
        if win_prob >= PROBABILITY_THRESHOLD:
            lot_size = math.floor((ACCOUNT_EQUITY * RISK_PCT) / (risk_dist * 100) * 100) / 100.0
            lot_size = max(0.01, lot_size)

            alert_msg = f"""🚨 *SIGNAL ALERT: XAU/USD ({signal_type})*
━━━━━━━━━━━━━━━━━━━━
🤖 *ML Win Probability:* `{win_prob*100:.1f}%` (Passed Filter)
📍 *Entry Price:* `${entry_price:,.2f}`
🛑 *Stop Loss (SL):* `${sl_price:,.2f}` (`${risk_dist:.2f}` Risk)
🎯 *Take Profit (TP):* `${tp_price:,.2f}` (R:R 1:{RR_RATIO})
⚖️ *Recommended Lot:* `{lot_size}` Lot (Risk 1%)
⏰ *Time:* `{latest_time.strftime('%Y-%m-%d %H:%M')}`
━━━━━━━━━━━━━━━━━━━━
💡 _ยกระดับด้วย H1-M15-M5 SMC + ML Filter_"""

            send_telegram_alert(alert_msg)

    except Exception as e:
        print(f"⚠️ เกิดข้อผิดพลาดขณะสแกนตลาด: {e}")

if __name__ == "__main__":
    print("🚀 เริ่มสแกนราคาทองคำสำหรับ Cloud Run / GitHub Actions...")
    scan_gold_market()
    print("🏁 สแกนเสร็จสิ้น ปิดการทำงาน Job")
