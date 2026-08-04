import os
import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

TWELVE_DATA_API_KEY = os.environ.get("TWELVE_DATA_API_KEY")
MODEL_FILE = "gold_ml_filter.pkl"


def fetch_historical_m5_data(count=500):
  """ดึงข้อมูลราคาย้อนหลัง M5 สำหรับ XAU/USD"""
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
    res = requests.get(url, params=params)
    data = res.json()
    if "values" not in data:
      print("ไม่สามารถดึงข้อมูลได้:", data.get("message"))
      return pd.DataFrame()

    parsed = []
    for c in reversed(data["values"]):
      parsed.append({
          "time": pd.to_datetime(c["datetime"]),
          "open": float(c["open"]),
          "high": float(c["high"]),
          "low": float(c["low"]),
          "close": float(c["close"]),
      })
    df = pd.DataFrame(parsed).set_index("time")
    return df
  except Exception as e:
    print(f"เกิดข้อผิดพลาด: {e}")
    return pd.DataFrame()


def prepare_dataset(df):
  """คำนวณ Indicators และสร้าง Label สำหรับการเทรน"""
  if len(df) < 50:
    return None, None

  # คำนวณ Features
  df["atr"] = (df["high"] - df["low"]).rolling(14).mean()
  delta = df["close"].diff()
  gain = (delta.where(delta > 0, 0)).rolling(14).mean()
  loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
  rs = gain / (loss.replace(0, 1e-6))
  df["rsi"] = 100 - (100 / (1 + rs))

  features = []
  labels = []

  for i in range(2, len(df) - 10):
    c1 = df.iloc[i - 2]
    c2 = df.iloc[i - 1]
    c3 = df.iloc[i]

    is_buy = c3["low"] > c1["high"]
    is_sell = c3["high"] < c1["low"]

    if not (is_buy or is_sell):
      continue

    fvg_size = (c3["low"] - c1["high"]) if is_buy else (c1["low"] - c3["high"])
    entry = c3["close"]
    sl = (c1["low"] - 1.5) if is_buy else (c1["high"] + 1.5)
    tp = (
        entry + (2.0 * abs(entry - sl))
        if is_buy
        else entry - (2.0 * abs(entry - sl))
    )

    t = df.index[i]
    hour = t.hour
    dayofweek = t.dayofweek
    risk_dist = abs(entry - sl)
    atr = df["atr"].iloc[i]
    rsi = df["rsi"].iloc[i]

    if pd.isna(atr) or pd.isna(rsi):
      continue

    # ตรวจสอบว่าในอีก 10 แท่งถัดไป ชน TP หรือ SL ก่อนกัน
    future_candles = df.iloc[i + 1 : i + 11]
    target = 0
    for _, fc in future_candles.iterrows():
      if is_buy:
        if fc["high"] >= tp:
          target = 1
          break
        if fc["low"] <= sl:
          target = 0
          break
      else:
        if fc["low"] <= tp:
          target = 1
          break
        if fc["high"] >= sl:
          target = 0
          break

    features.append([fvg_size, atr, rsi, hour, dayofweek, risk_dist])
    labels.append(target)

  return np.array(features), np.array(labels)


def train_and_save_model():
  print("กำลังดึงข้อมูลเพื่อเทรนโมเดล...")
  df = fetch_historical_m5_data(500)
  if df.empty:
    print("ไม่มีข้อมูลสำหรับเทรน")
    return

  X, y = prepare_dataset(df)
  if X is None or len(X) < 10:
    print("ข้อมูลสำหรับเทรนมีน้อยเกินไป")
    return

  print(
      f"ขนาด Dataset: {len(X)} ตัวอย่าง (Win: {sum(y)}, Loss: {len(y) - sum(y)})"
  )

  # สร้างและฝึกโมเดล Random Forest
  clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
  clf.fit(X, y)

  # บันทึกโมเดล
  joblib.dump(clf, MODEL_FILE)
  print(f"✨ บันทึกโมเดลเรียบร้อยแล้วลงไฟล์ '{MODEL_FILE}'")


if __name__ == "__main__":
  train_and_save_model()
    
