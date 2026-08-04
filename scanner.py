import json
import os
from datetime import datetime
import joblib
import gspread
from google.oauth2.service_account import Credentials
import numpy as np
import pandas as pd
import requests

# โหลดค่าตัวแปรสภาพแวดล้อม (Environment Variables)
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
TWELVE_DATA_API_KEY = os.environ.get("TWELVE_DATA_API_KEY")
GOOGLE_CREDENTIALS_JSON = os.environ.get("GOOGLE_CREDENTIALS_JSON")
GOOGLE_SHEET_NAME = os.environ.get("GOOGLE_SHEET_NAME", "Gold_Trading_Logs")

STATE_FILE = "last_signal.json"
MODEL_FILE = "gold_ml_filter.pkl"


def fetch_twelvedata_m5_data(count=100):
  """ดึงข้อมูลราคาย้อนหลัง M5 สำหรับ XAU/USD จาก Twelve Data API"""
  api_key = TWELVE_DATA_API_KEY
  symbol = "XAU/USD"
  interval = "5min"
  url = "https://api.twelvedata.com/time_series"
  params = {
      "symbol": symbol,
      "interval": interval,
      "outputsize": count,
      "apikey": api_key,
      "format": "JSON",
  }
  try:
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    if "values" not in data:
      print(f"Twelve Data Error: {data.get('message', 'Unknown error')}")
      return pd.DataFrame()

    values = data["values"]
    parsed_data = []
    # เรียงลำดับข้อมูลจากอดีตมาปัจจุบันเพื่อให้ Indicator คำนวณถูกต้อง
    for c in reversed(values):
      parsed_data.append({
          "time": pd.to_datetime(c["datetime"]),
          "open": float(c["open"]),
          "high": float(c["high"]),
          "low": float(c["low"]),
          "close": float(c["close"]),
          "volume": float(c["volume"])
          if "volume" in c and c["volume"] is not None
          else 0.0,
      })
    df = pd.DataFrame(parsed_data)
    if not df.empty:
      df.set_index("time", inplace=True)
    return df
  except Exception as e:
    print(f"เกิดข้อผิดพลาดในการดึงข้อมูลราคา: {e}")
    return pd.DataFrame()


def send_telegram_alert(message):
  """ส่งข้อความแจ้งเตือนเข้า Telegram"""
  if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    print("ไม่พบข้อมูล Telegram Credentials")
    return
  url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
  payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
  try:
    requests.post(url, json=payload)
  except Exception as e:
    print(f"เกิดข้อผิดพลาดในการส่ง Telegram: {e}")


def log_to_google_sheet(row_data):
  """บันทึกข้อมูลสัญญาณลง Google Sheets

  พร้อมตรวจสอบและสร้าง Worksheet และ Header อัตโนมัติ
  """
  if not GOOGLE_CREDENTIALS_JSON:
    print("ไม่พบข้อมูล Google Credentials JSON")
    return

  try:
    creds_dict = json.loads(GOOGLE_CREDENTIALS_JSON)
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
    client = gspread.authorize(creds)

    # 1. เปิด Google Spreadsheet ตามชื่อที่กำหนด
    try:
      spreadsheet = client.open(GOOGLE_SHEET_NAME)
    except gspread.exceptions.SpreadsheetNotFound:
      print(
          f"❌ ไม่พบ Spreadsheet ชื่อ '{GOOGLE_SHEET_NAME}'"
          " กรุณาสร้างไฟล์ Google Sheets และแชร์สิทธิ์ให้ Service Account"
          " เรียบร้อยแล้ว"
      )
      return

    # 2. ตรวจสอบว่ามี Worksheet ชื่อ "Signals" หรือยัง ถ้ายังให้สร้างอัตโนมัติ
    try:
      sheet = spreadsheet.worksheet("Signals")
    except gspread.exceptions.WorksheetNotFound:
      sheet = spreadsheet.add_worksheet(title="Signals", rows="1000", cols="10")
      print('✨ สร้าง Worksheet "Signals" ให้อัตโนมัติสำเร็จ')

    # 3. ตรวจสอบว่ามี Header หรือยัง ถ้าตารางว่าง ให้เพิ่ม Header อัตโนมัติทันที
    existing_data = sheet.get_all_values()
    if not existing_data:
      headers = ["Time", "Type", "Entry", "SL", "TP", "Status", "PnL"]
      sheet.append_row(headers)
      print("✨ เพิ่มตาราง Header อัตโนมัติสำเร็จ (Time, Type, Entry, SL, TP, Status, PnL)")

    # 4. บันทึกข้อมูลแถวใหม่ (Row Data) ลงในตาราง
    sheet.append_row(row_data)
    print("📥 บันทึกข้อมูลสัญญาณลง Google Sheets สำเร็จ")

  except Exception as e:
    print(f"⚠️ เกิดข้อผิดพลาดในการจัดการ Google Sheets: {e}")



def load_state():
  """โหลดสถานะเวลาของสัญญาณล่าสุดเพื่อป้องกันการแจ้งเตือนซ้ำ"""
  if os.path.exists(STATE_FILE):
    with open(STATE_FILE, "r") as f:
      return json.load(f)
  return {"last_time": ""}


def save_state(last_time):
  """บันทึกสถานะเวลาของสัญญาณล่าสุด"""
  with open(STATE_FILE, "w") as f:
    json.dump({"last_time": str(last_time)}, f)


def detect_smc_fvg(df):
  """ตรวจสอบโครงสร้าง Fair Value Gap (FVG) แบบ SMC"""
  if len(df) < 3:
    return None

  i = len(df) - 2  # แท่งเทียนที่ปิดสมบูรณ์แล้วแท่งล่าสุด
  if i < 2:
    return None

  c1 = df.iloc[i - 2]
  c2 = df.iloc[i - 1]
  c3 = df.iloc[i]

  # Bullish FVG
  if c3["low"] > c1["high"]:
    fvg_size = c3["low"] - c1["high"]
    return {
        "type": "BUY",
        "time": str(df.index[i]),
        "entry": c3["close"],
        "fvg_size": fvg_size,
        "sl": c1["low"] - 1.5,
    }

  # Bearish FVG
  elif c3["high"] < c1["low"]:
    fvg_size = c1["low"] - c3["high"]
    return {
        "type": "SELL",
        "time": str(df.index[i]),
        "entry": c3["close"],
        "fvg_size": fvg_size,
        "sl": c1["high"] + 1.5,
    }

  return None


def main():
  print("กำลังรันระบบ Gold SMC Scanner...")
  df = fetch_twelvedata_m5_data(100)
  if df.empty:
    print("ไม่สามารถดึงข้อมูลราคาได้")
    return

  signal = detect_smc_fvg(df)
  if not signal:
    print("ไม่พบสัญญาณ FVG ในรอบนี้")
    return

  state = load_state()
  if state.get("last_time") == signal["time"]:
    print("สัญญาณนี้ถูกส่งแจ้งเตือนไปแล้ว ข้ามการทำงาน")
    return

  # ตรวจสอบตัวกรอง Machine Learning (ถ้ามีไฟล์โมเดล)
  ml_passed = True
  if os.path.exists(MODEL_FILE):
    try:
      model = joblib.load(MODEL_FILE)
      close_prices = df["close"]
      atr = (df["high"] - df["low"]).rolling(14).mean().iloc[-1]
      delta = close_prices.diff()
      gain = (delta.where(delta > 0, 0)).rolling(14).mean()
      loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
      rs = gain / loss
      rsi = 100 - (100 / (1 + rs)).iloc[-1]

      t = pd.to_datetime(signal["time"])
      hour = t.hour
      dayofweek = t.dayofweek
      risk_dist = abs(signal["entry"] - signal["sl"])

      features = np.array(
          [[signal["fvg_size"], atr, rsi, hour, dayofweek, risk_dist]]
      )
      pred = model.predict(features)[0]
      if pred == 0:
        ml_passed = False
        print("สัญญาณถูกกรองออกโดยโมเดล Machine Learning")
    except Exception as e:
      print(f"เกิดข้อผิดพลาดในระบบ ML Filter: {e}")

  if ml_passed:
    msg = (
        f"🚨 *Gold SMC Signal Detected!*\n"
        f"Type: *{signal['type']}*\n"
        f"Time: `{signal['time']}`\n"
        f"Entry: `{signal['entry']:.2f}`\n"
        f"SL: `{signal['sl']:.2f}`\n"
        f"FVG Size: `{signal['fvg_size']:.2f}`"
    )
    send_telegram_alert(msg)

    tp = (
        signal["entry"] + (2.0 * abs(signal["entry"] - signal["sl"]))
        if signal["type"] == "BUY"
        else signal["entry"] - (2.0 * abs(signal["entry"] - signal["sl"]))
    )
    row_data = [
        signal["time"],
        signal["type"],
        signal["entry"],
        signal["sl"],
        tp,
        "OPEN",
        0.0,
    ]
    log_to_google_sheet(row_data)
    save_state(signal["time"])
    print("ส่งสัญญาณและบันทึกข้อมูลเรียบร้อยแล้ว")


if __name__ == "__main__":
  main()
 
