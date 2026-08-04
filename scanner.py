import os
import json
import requests
import yfinance as yf
import pandas as pd
import numpy as np

# อ่านค่า Secret จาก Environment Variables
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# โหลด Config
CONFIG_FILE = "best_config.json"
if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "r") as f:
        config = json.load(f)
else:
    config = {"PROBA_THRESHOLD": 0.55, "RR_RATIO": 1.5, "GOLD_SL_BUFFER": 2.5}

def send_telegram_msg(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram Token หรือ Chat ID ไม่ถูกตั้งค่า")
        return
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        res = requests.post(url, json=payload)
        if res.status_code == 200:
            print("✅ ส่งข้อความเข้า Telegram เรียบร้อยแล้ว")
        else:
            print(f"❌ ส่ง Telegram ไม่สำเร็จ: {res.text}")
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการส่ง Telegram: {e}")

def run_scanner():
    print("🔍 กำลังเริ่มสแกนราคาทองคำ (GC=F)...")
    ticker = yf.Ticker("GC=F")
    df = ticker.history(period="5d", interval="5m")
    
    if df.empty:
        print("❌ ไม่สามารถดึงข้อมูลราคาได้")
        return

    # คำนวณอินดิเคเตอร์พื้นฐาน
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    last_row = df.iloc[-1]
    close_price = last_row['Close']
    sma20 = last_row['SMA20']

    print(f"📈 ราคาปัจจุบัน: {close_price:.2f} | SMA20: {sma20:.2f}")

    # ตัวอย่างการตรวจจับสัญญาณ (ปรับเงื่อนไขตามกลยุทธ์ของคุณ)
    # เช่น ราคาตัดขึ้นเหนือ SMA20 = BUY
    if close_price > sma20:
        sl = close_price - config['GOLD_SL_BUFFER']
        tp = close_price + (config['GOLD_SL_BUFFER'] * config['RR_RATIO'])
        
        msg = (
            f"🚀 *GOLD SIGNAL (M5)*\n"
            f"-------------------------\n"
            f"🟢 *Action:* BUY\n"
            f"💰 *Price:* {close_price:.2f}\n"
            f"🛑 *SL:* {sl:.2f}\n"
            f"🎯 *TP:* {tp:.2f}\n"
            f"⚙️ *RR:* {config['RR_RATIO']} | *SL Buffer:* {config['GOLD_SL_BUFFER']}"
        )
        send_telegram_msg(msg)
    else:
        print("ℹ️ ยังไม่มีสัญญาณเข้าเทรดในแท่งนี้")

if __name__ == "__main__":
    run_scanner()
