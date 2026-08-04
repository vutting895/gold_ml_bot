import os
import joblib
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# ==================== CONFIGURATION ====================
SYMBOL = "GC=F"                  # Spot Gold / Gold Futures
MODEL_OUTPUT_PATH = "gold_ml_filter.pkl"
RR_RATIO = 3.0                   # Target Risk-to-Reward Ratio (1:3)
GOLD_SL_BUFFER = 0.80            # SL Buffer ($0.80)

def create_training_dataset():
    print("📥 1/5. กำลังดึงข้อมูลราคาทองคำ M5 ย้อนหลัง...")
    df_5m = yf.download(SYMBOL, period="60d", interval="5m", progress=False)

    if isinstance(df_5m.columns, pd.MultiIndex):
        df_5m.columns = df_5m.columns.get_level_values(0)

    df_5m = df_5m.dropna()
    print(f"📊 โหลดข้อมูลได้ทั้งหมด {len(df_5m)} แท่ง")

    print("⚙️ 2/5. คำนวณ Indicators และ Multi-Timeframe Context...")
    # Resample เป็น M15 และ H1
    df_15m = df_5m.resample('15min').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last'}).dropna()
    df_1h  = df_5m.resample('1H').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last'}).dropna()

    # EMA Trends
    df_1h['EMA_50'] = df_1h['Close'].ewm(span=50, adjust=False).mean()
    df_1h['H1_Trend'] = np.where(df_1h['Close'] > df_1h['EMA_50'], 1, -1)

    df_15m['EMA_20'] = df_15m['Close'].ewm(span=20, adjust=False).mean()
    df_15m['M15_Trend'] = np.where(df_15m['Close'] > df_15m['EMA_20'], 1, -1)

    df_5m['H1_Trend'] = df_1h['H1_Trend'].reindex(df_5m.index, method='ffill')
    df_5m['M15_Trend'] = df_15m['M15_Trend'].reindex(df_5m.index, method='ffill')

    # ATR 14
    high_low = df_5m['High'] - df_5m['Low']
    high_cp  = np.abs(df_5m['High'] - df_5m['Close'].shift(1))
    low_cp   = np.abs(df_5m['Low'] - df_5m['Close'].shift(1))
    df_5m['ATR'] = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1).rolling(14).mean()

    # Fair Value Gap (FVG)
    df_5m['Bullish_FVG'] = df_5m['Low'] > df_5m['High'].shift(2)
    df_5m['Bearish_FVG'] = df_5m['High'] < df_5m['Low'].shift(2)

    # 3/5. การสกัด Feature และการทำ Labeling (Is_Win)
    print("🏷️ 3/5. สกัด Feature และสร้าง Label การชนะ/แพ้ (Forward Simulation)...")
    dataset = []

    for i in range(50, len(df_5m) - 150):
        latest_bar  = df_5m.iloc[i]
        prev_bar    = df_5m.iloc[i-1]
        latest_time = df_5m.index[i]

        is_long  = (latest_bar['H1_Trend'] == 1) and (latest_bar['M15_Trend'] == 1) and latest_bar['Bullish_FVG']
        is_short = (latest_bar['H1_Trend'] == -1) and (latest_bar['M15_Trend'] == -1) and latest_bar['Bearish_FVG']

        if not (is_long or is_short):
            continue

        entry_price = float(latest_bar['Close'])
        if is_long:
            sl_price  = float(min(latest_bar['Low'], prev_bar['Low'])) - GOLD_SL_BUFFER
            risk_dist = entry_price - sl_price
            tp_price  = entry_price + (risk_dist * RR_RATIO)
            fvg_size  = float(latest_bar['Low'] - df_5m.iloc[i-2]['High'])
        else:
            sl_price  = float(max(latest_bar['High'], prev_bar['High'])) + GOLD_SL_BUFFER
            risk_dist = sl_price - entry_price
            tp_price  = entry_price - (risk_dist * RR_RATIO)
            fvg_size  = float(df_5m.iloc[i-2]['Low'] - latest_bar['High'])

        if risk_dist <= 0:
            continue

        # ตรวจสอบอนาคต 150 แท่งว่าชน TP (1) หรือ SL (0) ก่อน
        future_bars = df_5m.iloc[i+1 : i+150]
        is_win = None

        for _, f_bar in future_bars.iterrows():
            if is_long:
                if f_bar['High'] >= tp_price:
                    is_win = 1
                    break
                elif f_bar['Low'] <= sl_price:
                    is_win = 0
                    break
            else:
                if f_bar['Low'] <= tp_price:
                    is_win = 1
                    break
                elif f_bar['High'] >= sl_price:
                    is_win = 0
                    break

        if is_win is not None:
            dataset.append({
                'FVG_Size': fvg_size,
                'ATR': float(latest_bar['ATR']),
                'Hour': int(latest_time.hour),
                'Minute': int(latest_time.minute),
                'Risk_Distance': risk_dist,
                'Target_IsWin': is_win
            })

    data_df = pd.DataFrame(dataset).dropna()
    print(f"✅ สกัดชุดข้อมูลสำเร็จ: พบทั้งหมด {len(data_df)} ตัวอย่าง (Win: {data_df['Target_IsWin'].sum()} / Loss: {len(data_df) - data_df['Target_IsWin'].sum()})")
    return data_df

def train_and_save_model():
    df = create_training_dataset()

    if len(df) < 50:
        print("❌ ตัวอย่างข้อมูลมีน้อยเกินไปสำหรับการเทรน")
        return

    # แยก Features (X) และ Target (y)
    feature_cols = ['FVG_Size', 'ATR', 'Hour', 'Minute', 'Risk_Distance']
    X = df[feature_cols]
    y = df['Target_IsWin']

    # แบ่งข้อมูล Train / Test Set (80 / 20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("\n🧠 4/5. กำลังเริ่มเทรนโมเดล Random Forest...")
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)

    # 5/5. ประเมินผลโมเดล
    print("\n📊 5/5. ประเมินประสิทธิภาพโมเดลบน Test Set:")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("--------------------------------------------------")
    print(classification_report(y_test, y_pred, target_names=['Loss (0)', 'Win (1)']))
    print(f"📈 ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
    print("--------------------------------------------------")

    # บันทึกไฟล์โมเดล
    joblib.dump(model, MODEL_OUTPUT_PATH)
    print(f"🎉 บันทึกไฟล์โมเดลเรียบร้อยแล้วที่: {MODEL_OUTPUT_PATH}")

if __name__ == "__main__":
    train_and_save_model()
  
