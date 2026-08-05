import json
import os
from datetime import datetime
import joblib
import gspread
from google.oauth2.service_account import Credentials
import numpy as np
import pandas as pd
import pytz
import requests

# โหลดค่าตัวแปรสภาพแวดล้อม (Environment Variables)
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
TWELVE_DATA_API_KEY = os.environ.get("TWELVE_DATA_API_KEY")
GOOGLE_CREDENTIALS_JSON = os.environ.get("GOOGLE_CREDENTIALS_JSON")
GOOGLE_SHEET_NAME = os.environ.get("GOOGLE_SHEET_NAME", "Gold_Trading_Logs")

MODEL_FILE = "gold_ml_filter.pkl"


# ==========================================
# 1. TELEGRAM NOTIFICATION SYSTEM (HTML)
# ==========================================
def send_telegram_message(message: str) -> bool:
  """ฟังก์ชันส่งข้อความไปยัง Telegram Bot ผ่าน HTTP POST (ใช้ HTML Format)"""
  if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    print(
        "❌ [Telegram Error] ไม่พบ TELEGRAM_BOT_TOKEN หรือ TELEGRAM_CHAT_ID"
    )
    return False

  url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
  payload = {
      "chat_id": TELEGRAM_CHAT_ID,
      "text": message,
      "parse_mode": "HTML",
      "disable_web_page_preview": True,
  }

  try:
    response = requests.post(url, json=payload, timeout=10)
    response.raise_for_status()
    print("✅ [Telegram] ส่งข้อความแจ้งเตือนสำเร็จ")
    return True
  except Exception as e:
    print(f"❌ [Telegram Error] ไม่สามารถส่งข้อความได้: {e}")
    return False


def send_signal_alert(
    symbol: str,
    signal_type: str,
    entry: float,
    sl: float,
    tp: float,
    timeframe: str = "M5",
    fvg_size: float = None,
) -> bool:
  """จัดรูปแบบการ์ดสัญญาณแจ้งเตือน SMC และส่งเข้า Telegram"""
  tz_th = pytz.timezone("Asia/Bangkok")
  now_th = datetime.now(tz_th).strftime("%Y-%m-%d %H:%M:%S")

  is_buy = signal_type.upper() == "BUY"
  type_emoji = "🟢 BUY" if is_buy else "🔴 SELL"
  trend_icon = "📈" if is_buy else "📉"

  risk_pips = abs(entry - sl)
  reward_pips = abs(tp - entry)
  rr_ratio = round(reward_pips / risk_pips, 2) if risk_pips > 0 else 0.0

  message = (
      f"🚨 <b>{symbol} {timeframe} SMC SIGNAL</b> {trend_icon}\n"
      f"━━━━━━━━━━━━━━━━━━\n"
      f"<b>Type:</b> {type_emoji}\n"
      f"<b>Time (TH):</b> <code>{now_th}</code> (UTC+7)\n\n"
      f"🎯 <b>Entry Price:</b> <code>${entry:.2f}</code>\n"
      f"🛑 <b>Stop Loss (SL):</b> <code>${sl:.2f}</code> (Risk:"
      f" ${risk_pips:.2f})\n"
      f"🏁 <b>Take Profit (TP):</b> <code>${tp:.2f}</code> (Reward:"
      f" ${reward_pips:.2f})\n"
      f"⚖️ <b>Risk : Reward:</b> <code>1:{rr_ratio}</code>\n"
  )

  if fvg_size:
    message += f"📐 <b>FVG Gap Size:</b> <code>${fvg_size:.2f}</code>\n"

  message += (
      "━━━━━━━━━━━━━━━━━━\n"
      "🤖 <i>Validated by ML Filter & Automated System</i>\n"
  )

  return send_telegram_message(message)


# ==========================================
# 2. DATA FETCHING & GOOGLE SHEETS SYSTEM
# ==========================================
def fetch_twelvedata_m5_data(count=100):
  """ดึงข้อมูลราคาย้อนหลัง M5 สำหรับ XAU/USD จาก Twelve Data API (โซนเวลาไทย Asia/Bangkok UTC+7)"""
  url = "https://api.twelvedata.com/time_series"
  params = {
      "symbol": "XAU/USD",
      "interval": "5min",
      "outputsize": count,
      "apikey": TWELVE_DATA_API_KEY,
      "format": "JSON",
      "timezone": "Asia/Bangkok",
  }
  try:
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()
    if "values" not in data:
      print(f"Twelve Data Error: {data.get('message', 'Unknown error')}")
      return pd.DataFrame()

    parsed_data = []
    for c in reversed(data["values"]):
      dt = pd.to_datetime(c["datetime"])
      parsed_data.append({
          "time": dt,
          "open": float(c["open"]),
          "high": float(c["high"]),
          "low": float(c["low"]),
          "close": float(c["close"]),
          "volume": float(c["volume"]) if c.get("volume") is not None else 0.0,
      })
    df = pd.DataFrame(parsed_data)
    if not df.empty:
      df.set_index("time", inplace=True)
    return df
  except Exception as e:
    print(f"เกิดข้อผิดพลาดในการดึงข้อมูลราคา: {e}")
    return pd.DataFrame()


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

    try:
      sheet = spreadsheet.worksheet("Signals")
    except gspread.exceptions.WorksheetNotFound:
      sheet = spreadsheet.add_worksheet(title="Signals", rows="1000", cols="10")

    existing_data = sheet.get_all_values()
    if not existing_data:
      headers = ["Time (UTC+7)", "Type", "Entry", "SL", "TP", "Status", "PnL"]
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
    if len(records) > 1:
      return records[-1][0]
  except Exception as e:
    print(f"เกิดข้อผิดพลาดในการเช็คเวลาสัญญาณล่าสุด: {e}")
  return ""


def update_open_trades_status(sheet, df_prices):
  """ตรวจสอบและอัปเดตออเดอร์สถานะ OPEN ใน Google Sheets ว่าชน TP หรือ SL หรือยัง"""
  if not sheet or df_prices.empty:
    return

  try:
    records = sheet.get_all_records()
    if not records:
      return

    print("🔍 กำลังตรวจสอบออเดอร์ที่เปิดค้างไว้ (OPEN)...")
    tz_th = pytz.timezone("Asia/Bangkok")

    for idx, row in enumerate(records, start=2):  # Row 1 คือ Header
      if str(row.get("Status", "")).upper() == "OPEN":
        entry = float(row["Entry"])
        sl = float(row["SL"])
        tp = float(row["TP"])
        trade_type = str(row["Type"]).upper()

        # แปลงเวลาของสัญญาณเป็น timezone-aware (Asia/Bangkok)
        signal_time = pd.to_datetime(row["Time (UTC+7)"])
        if signal_time.tzinfo is None:
          signal_time = tz_th.localize(signal_time)

        # กรองแท่งเทียนที่เกิดหลังจากสัญญาณ
        df_index_tz = (
            df_prices.index.tz_localize(tz_th)
            if df_prices.index.tzinfo is None
            else df_prices.index
        )
        future_candles = df_prices[df_index_tz > signal_time]

        for _, candle in future_candles.iterrows():
          high = candle["high"]
          low = candle["low"]

          # กรณี BUY Order
          if trade_type == "BUY":
            if low <= sl:  # ชน Stop Loss
              pnl = round(sl - entry, 2)
              sheet.update_cell(idx, 6, "LOSS")
              sheet.update_cell(idx, 7, pnl)
              print(
                  f"❌ Order BUY เวลา {row['Time (UTC+7)']} ชน SL (PnL:"
                  f" ${pnl})"
              )
              break
            elif high >= tp:  # ชน Take Profit
              pnl = round(tp - entry, 2)
              sheet.update_cell(idx, 6, "WIN")
              sheet.update_cell(idx, 7, pnl)
              print(
                  f"🎯 Order BUY เวลา {row['Time (UTC+7)']} ชน TP (PnL:"
                  f" ${pnl})"
              )
              break

          # กรณี SELL Order
          elif trade_type == "SELL":
            if high >= sl:  # ชน Stop Loss
              pnl = round(entry - sl, 2)
              sheet.update_cell(idx, 6, "LOSS")
              sheet.update_cell(idx, 7, pnl)
              print(
                  f"❌ Order SELL เวลา {row['Time (UTC+7)']} ชน SL (PnL:"
                  f" ${pnl})"
              )
              break
            elif low <= tp:  # ชน Take Profit
              pnl = round(entry - tp, 2)
              sheet.update_cell(idx, 6, "WIN")
              sheet.update_cell(idx, 7, pnl)
              print(
                  f"🎯 Order SELL เวลา {row['Time (UTC+7)']} ชน TP (PnL:"
                  f" ${pnl})"
              )
              break

  except Exception as e:
    print(f"⚠️ เกิดข้อผิดพลาดในการอัปเดตสถานะออเดอร์: {e}")


# ==========================================
# 3. SMC DETECTOR & MAIN SCANNER LOGIC
# ==========================================
def detect_smc_fvg(df):
  """ตรวจสอบโครงสร้าง Fair Value Gap (FVG) แบบ SMC"""
  if len(df) < 3:
    return None

  i = len(df) - 2
  if i < 2:
    return None

  c1 = df.iloc[i - 2]
  c2 = df.iloc[i - 1]
  c3 = df.iloc[i]

  time_str = df.index[i].strftime("%Y-%m-%d %H:%M:%S")

  # Bullish FVG
  if c3["low"] > c1["high"]:
    fvg_size = c3["low"] - c1["high"]
    return {
        "type": "BUY",
        "time": time_str,
        "entry": c3["close"],
        "fvg_size": fvg_size,
        "sl": c1["low"] - 1.5,
    }

  # Bearish FVG
  elif c3["high"] < c1["low"]:
    fvg_size = c1["low"] - c3["high"]
    return {
        "type": "SELL",
        "time": time_str,
        "entry": c3["close"],
        "fvg_size": fvg_size,
        "sl": c1["high"] + 1.5,
    }

  return None


def main():
  print("กำลังรันระบบ Gold SMC Scanner บน GitHub Actions (โซนเวลาไทย UTC+7)...")

  # 1. เช็กวันเสาร์ - อาทิตย์ (ตลาดทองคำปิด)
  tz_th = pytz.timezone("Asia/Bangkok")
  now_th = datetime.now(tz_th)
  if now_th.weekday() in [5, 6]:
    print("😴 ตลาดทองคำปิดทำการ (วันเสาร์-อาทิตย์) ข้ามการสแกน")
    return

  sheet = get_google_sheet_handle()

  df = fetch_twelvedata_m5_data(100)
  if df.empty:
    print("ไม่สามารถดึงข้อมูลราคาได้")
    return

  # 2. ตรวจสอบออเดอร์เก่าสถานะ OPEN ใน Google Sheets
  if sheet:
    update_open_trades_status(sheet, df)

  # 3. ตรวจจับสัญญาณ SMC FVG ใหม่
  signal = detect_smc_fvg(df)
  if not signal:
    print("ไม่พบสัญญาณ FVG ในรอบนี้")
    return

  last_time_in_sheet = get_last_signal_time(sheet)
  if last_time_in_sheet == signal["time"]:
    print(
        f"สัญญาณเวลา {signal['time']} (UTC+7) ถูกแจ้งเตือนไปแล้ว ข้ามการทำงาน"
    )
    return

  # 4. ตรวจสอบผ่าน Machine Learning Model
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
      rsi = (100 - (100 / (1 + rs))).iloc[-1]

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
    # คำนวณ Take Profit (RR 1:2)
    risk = abs(signal["entry"] - signal["sl"])
    tp = (
        signal["entry"] + (2.0 * risk)
        if signal["type"] == "BUY"
        else signal["entry"] - (2.0 * risk)
    )

    # 5. ส่งสัญญาณเข้า Telegram
    send_signal_alert(
        symbol="XAU/USD",
        signal_type=signal["type"],
        entry=signal["entry"],
        sl=signal["sl"],
        tp=tp,
        timeframe="M5",
        fvg_size=signal.get("fvg_size"),
    )

    # 6. บันทึกลง Google Sheets
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

    print(f"ส่งสัญญาณเวลา {signal['time']} (UTC+7) เรียบร้อยแล้ว")


if __name__ == "__main__":
  main()
              
