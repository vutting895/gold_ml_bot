"""
Gold ML Model Trainer (train_gold_model.py)
อัปเกรด Advanced Feature Engineering (RSI, DayOfWeek)
และใช้ class_weight='balanced' ใน RandomForestClassifier
"""

import os
import joblib
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

SYMBOL = "GC=F"
MODEL_OUTPUT_PATH = "gold_ml_filter.pkl"
LOOKBACK_PERIOD = "60d"
RR_RATIO = 3.0
SL_BUFFER = 0.80

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def fetch_and_prepare_data():
    print(f"📥 กำลังดึงข้อมูลราคาทองคำ ({SYMBOL}) ย้อนหลัง {LOOKBACK_PERIOD}...")
    df_5m = yf.download(SYMBOL, period=LOOKBACK_PERIOD, interval="5m", progress=False)

    if isinstance(df_5m.columns, pd.MultiIndex):
        df_5m.columns = df_5m.columns.get_level_values(0)

    df_5m = df_5m.dropna()
    if len(df_5m) < 200:
        return None

    # Resampling
    df_15m = df_5m.resample('15min').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last'}).dropna()
    df_1h  = df_5m.resample('1H').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last'}).dropna()

    df_1h['EMA_50'] = df_1h['Close'].ewm(span=50, adjust=False).mean()
    df_1h['H1_Trend'] = np.where(df_1h['Close'] > df_1h['EMA_50'], 1, -1)

    df_15m['EMA_20'] = df_15m['Close'].ewm(span=20, adjust=False).mean()
    df_15m['M15_Trend'] = np.where(df_15m['Close'] > df_15m['EMA_20'], 1, -1)

    df_5m['H1_Trend'] = df_1h['H1_Trend'].reindex(df_5m.index, method='ffill')
    df_5m['M15_Trend'] = df_15m['M15_Trend'].reindex(df_5m.index, method='ffill')

    # Priority 3 Features: ATR & RSI
    high_low = df_5m['High'] - df_5m['Low']
    high_cp  = np.abs(df_5m['High'] - df_5m['Close'].shift(1))
    low_cp   = np.abs(df_5m['Low'] - df_5m['Close'].shift(1))
    df_5m['ATR'] = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1).rolling(14).mean()
    df_5m['RSI'] = calculate_rsi(df_5m['Close'], 14)

    df_5m['Bullish_FVG'] = df_5m['Low'] > df_5m['High'].shift(2)
    df_5m['Bearish_FVG'] = df_5m['High'] < df_5m['Low'].shift(2)

    return df_5m.dropna()

def create_dataset(df):
    samples = []
    for i in range(50, len(df) - 100):
        latest_bar  = df.iloc[i]
        prev_bar    = df.iloc[i-1]
        prev_2_bar  = df.iloc[i-2]
        latest_time = df.index[i]

        is_long  = (latest_bar['H1_Trend'] == 1) and (latest_bar['M15_Trend'] == 1) and latest_bar['Bullish_FVG']
        is_short = (latest_bar['H1_Trend'] == -1) and (latest_bar['M15_Trend'] == -1) and latest_bar['Bearish_FVG']

        if not (is_long or is_short):
            continue

        entry_price = float(latest_bar['Close'])

        if is_long:
            sl_price  = float(min(latest_bar['Low'], prev_bar['Low'])) - SL_BUFFER
            risk_dist = entry_price - sl_price
            tp_price  = entry_price + (risk_dist * RR_RATIO)
            fvg_size  = float(latest_bar['Low'] - prev_2_bar['High'])
        else:
            sl_price  = float(max(latest_bar['High'], prev_bar['High'])) + SL_BUFFER
            risk_dist = sl_price - entry_price
            tp_price  = entry_price - (risk_dist * RR_RATIO)
            fvg_size  = float(prev_2_bar['Low'] - latest_bar['High'])

        if risk_dist <= 0:
            continue

        future_bars = df.iloc[i+1 : i+101]
        label = 0

        for _, f_bar in future_bars.iterrows():
            if is_long:
                if f_bar['High'] >= tp_price:
                    label = 1
                    break
                if f_bar['Low'] <= sl_price:
                    label = 0
                    break
            else:
                if f_bar['Low'] <= tp_price:
                    label = 1
                    break
                if f_bar['High'] >= sl_price:
                    label = 0
                    break

        samples.append({
            'FVG_Size': fvg_size,
            'ATR': float(latest_bar['ATR']),
            'RSI': float(latest_bar['RSI']),
            'Hour': int(latest_time.hour),
            'DayOfWeek': int(latest_time.dayofweek),
            'Risk_Distance': risk_dist,
            'Target': label
        })

    return pd.DataFrame(samples)

def train_and_save_model():
    df = fetch_and_prepare_data()
    if df is None:
        return

    dataset = create_dataset(df)
    if dataset.empty or len(dataset) < 30:
        print("❌ ตัวอย่างข้อมูลมีจำนวนน้อยเกินไปสำหรับการเทรน")
        return

    feature_cols = ['FVG_Size', 'ATR', 'RSI', 'Hour', 'DayOfWeek', 'Risk_Distance']
    X = dataset[feature_cols]
    y = dataset['Target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n🧠 กำลังเทรนโมเดล Machine Learning (Random Forest Balanced)...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_split=5,
        class_weight='balanced',  # แก้ไข Class Imbalance
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n🎯 Model Accuracy บน Test Set: {acc*100:.2f}%")
    print(classification_report(y_test, y_pred, zero_division=0))

    joblib.dump(model, MODEL_OUTPUT_PATH)
    print(f"💾 บันทึกโมเดลสำเร็จเรียบร้อยที่ไฟล์: '{MODEL_OUTPUT_PATH}'")

if __name__ == "__main__":
    train_and_save_model()
    
