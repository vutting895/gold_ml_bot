import json
import os
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


def get_google_sheet_handle():
  """จัดการเชื่อมต่อและดึง Object ของ Worksheet 'Signals'"""
  if not GOOGLE_CREDENTIALS_JSON:
    print("ไม่พบข้อมูล Google Credentials JSON")
    return None

  try:
    creds_dict = json.loads(GOOGLE_CREDENTIALS_JSON)
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
    client = gspread.authorize(creds)

    spreadsheet = client.open(GOOGLE_SHEET_NAME)

    # ดึงหรือสร้าง Worksheet "Signals"
    try:
      sheet = spreadsheet.worksheet("Signals")
    except gspread.exceptions.WorksheetNotFound:
      sheet = spreadsheet.add_worksheet(title="Signals", rows="1000", cols="10")

    # ตรวจสอบว่ามี Header หรือยัง
    existing_data = sheet.get_all_values()
    if not existing_data:
      headers = ["Time", "Type", "Entry", "SL", "TP", "Status", "PnL"]
      sheet.append_row(headers)

    return sheet
  except Exception as e:
    print(f"⚠️ เกิดข้อผิดพลาดในการเชื่อมต่อ Google Sheets: {e}")
    return None


def get_last_signal_time(sheet):
  """ดึงเวลาของสัญญาณล่าสุดจาก Google Sheets เพื่อเช็คการส่งซ้ำ"""
  if not sheet:
    return ""
  try:
    records = sheet.get_all_values()
    if len(records) > 1:  # มีข้อมูลมากกว่าแค่ Header
      return records[-1][0]  # คอลัมน์แรก (Time)
  except Exception as e:
    print(f"เกิดข้อผิดพลาดในการเช็คเวลาสัญญาณล่าสุด: {e}")
  return ""


def detect_smc_fvg(df):
  """ตรวจสอบโครงสร้าง Fair Value Gap (FVG) แบบ SMC"""
  if len(df) < 3:
    return None

  i = len(df) - 2  # แท่งเทียนที่ปิดสมบูรณ์แล้ว
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
  print("กำลังรันระบบ Gold SMC Scanner บน GitHub Actions...")

  # 1. เชื่อมต่อ Google Sheets
  sheet = get_google_sheet_handle()

  # 2. ดึงข้อมูลราคา
  df = fetch_twelvedata_m5_data(100)
  if df.empty:
    print("ไม่สามารถดึงข้อมูลราคาได้")
    return

  # 3. ตรวจจับสัญญาณ SMC
  signal = detect_smc_fvg(df)
  if not signal:
    print("ไม่พบสัญญาณ FVG ในรอบนี้")
    return

  # 4. ตรวจสอบสัญญาณซ้ำจาก Google Sheets
  last_time_in_sheet = get_last_signal_time(sheet)
  if last_time_in_sheet == signal["time"]:
    print(f"สัญญาณเวลา {signal['time']} ถูกแจ้งเตือนไปแล้ว ข้ามการทำงาน")
    return

  # 5. กรองด้วย Machine Learning (ถ้ามีไฟล์โมเดล)
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

  # 6. ส่งแจ้งเตือนและบันทึกข้อมูล
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

    if sheet:
      sheet.append_row(row_data)
      print("บันทึกข้อมูลลง Google Sheets เรียบร้อยแล้ว")

    print("ส่งสัญญาณเรียบร้อยแล้ว")


if __name__ == "__main__":
  main()
  
