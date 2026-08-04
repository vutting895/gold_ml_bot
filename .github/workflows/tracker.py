"""
Trade Result Tracker (tracker.py)
สคริปต์ตรวจเช็กสถานะไม้ที่เปิดอยู่ (OPEN) ว่าชน TP / SL / BE แล้วหรือยัง
"""

import os
import json
import gspread
import yfinance as yf
from oauth2client.service_account import ServiceAccountCredentials

GOOGLE_SHEET_NAME = "Gold_Trading_Logs"

def update_trade_results():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds_json = os.getenv("GOOGLE_CREDENTIALS_JSON")
    if creds_json:
        creds = ServiceAccountCredentials.from_json_keyfile_dict(json.loads(creds_json), scope)
    elif os.path.exists("google_credentials.json"):
        creds = ServiceAccountCredentials.from_json_keyfile_dict(json.load(open("google_credentials.json")), scope)
    else:
        print("❌ ไม่พบ Credentials")
        return

    client = gspread.authorize(creds)
    sheet = client.open(GOOGLE_SHEET_NAME).worksheet("Signals")
    records = sheet.get_all_records()

    if not records:
        print("ℹ️ ไม่มีข้อมูลใน Sheet")
        return

    # ดึงราคาปัจจุบันจาก yfinance
    df = yf.download("GC=F", period="1d", interval="1m", progress=False)
    if df.empty:
        return
    
    current_price = float(df['Close'].iloc[-1])
    high_price    = float(df['High'].max())
    low_price     = float(df['Low'].min())

    for idx, row in enumerate(records, start=2): # start=2 เพราะแถว 1 คือ Header
        status = row.get("Status")
        if status != "OPEN":
            continue

        signal_type = row.get("Type")
        entry = float(row.get("Entry"))
        sl    = float(row.get("SL"))
        tp    = float(row.get("TP"))
        be    = float(row.get("BE_Trigger"))

        is_buy = "BUY" in signal_type

        # ตรวจสอบการชน TP / SL
        new_status = "OPEN"
        exit_price = 0.0
        pnl = 0.0

        if is_buy:
            if high_price >= tp:
                new_status = "WIN (TP)"
                exit_price = tp
                pnl = tp - entry
            elif low_price <= sl:
                new_status = "LOSS (SL)"
                exit_price = sl
                pnl = sl - entry
            elif high_price >= be and low_price <= (entry + 0.30):
                new_status = "BE (Break-Even)"
                exit_price = entry + 0.30
                pnl = 0.30
        else: # SELL
            if low_price <= tp:
                new_status = "WIN (TP)"
                exit_price = tp
                pnl = entry - tp
            elif high_price >= sl:
                new_status = "LOSS (SL)"
                exit_price = sl
                pnl = entry - sl
            elif low_price <= be and high_price >= (entry - 0.30):
                new_status = "BE (Break-Even)"
                exit_price = entry - 0.30
                pnl = 0.30

        # อัปเดตสถานะกลับลง Google Sheet
        if new_status != "OPEN":
            sheet.update_cell(idx, 8, new_status)   # คอลัมน์ Status
            sheet.update_cell(idx, 9, exit_price)   # คอลัมน์ Exit_Price
            sheet.update_cell(idx, 10, round(pnl, 2)) # คอลัมน์ PnL
            print(f"✅ อัปเดตสถานะแถวที่ {idx}: {new_status} (PnL: ${pnl:.2f})")

if __name__ == "__main__":
    update_trade_results()
          
