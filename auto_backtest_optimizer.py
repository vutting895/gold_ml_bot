"""
Auto Backtest & Parameter Optimizer (auto_backtest_optimizer.py)
ระบบจำลองการเทรดย้อนหลังเพื่อค้นหาพารามิเตอร์ที่ให้ Win Rate และ Profit Factor ดีที่สุด
จากนั้นบันทึกค่าลงไฟล์ best_config.json เพื่อให้ scanner.py ดึงไปใช้งาน
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# ==================== CONFIGURATION ====================
SYMBOL = "GC=F"                      # Spot Gold บน Yahoo Finance
MODEL_PATH = "gold_ml_filter.pkl"   # ไฟล์โมเดล ML
CONFIG_OUTPUT_PATH = "best_config.json"
LOOKBACK_PERIOD = "60d"             # ระยะเวลาที่ใช้ Backtest

# ขอบเขตพารามิเตอร์สำหรับทำ Grid Search
PARAM_GRID = {
    'PROBA_THRESHOLD': [0.55, 0.60, 0.65, 0.70],
    'RR_RATIO': [1.5, 2.0, 2.5, 3.0],
    'GOLD_SL_BUFFER': [0.50, 0.80, 1.00, 1.20]
}

def fetch_backtest_data():
    """ดึงข้อมูลราคา และคำนวณ Indicators สำหรับการทำ Backtest"""
    print(f"📥 กำลังดึงข้อมูลราคาทองคำ ({SYMBOL}) สำหรับ Backtest ย้อนหลัง {LOOKBACK_PERIOD}...")
    df_5m = yf.download(SYMBOL, period=LOOKBACK_PERIOD, interval="5m", progress=False)

    if isinstance(df_5m.columns, pd.MultiIndex):
        df_5m.columns = df_5m.columns.get_level_values(0)

    df_5m = df_5m.dropna()
    if len(df_5m) < 200:
        print("❌ ข้อมูลไม่เพียงพอสำหรับการทำ Backtest")
        return None

    # Multi-timeframe Resampling (H1 และ M15)
    df_15m = df_5m.resample('15min').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last'}).dropna()
    df_1h  = df_5m.resample('1H').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last'}).dropna()

    # คำนวณ Trend Indicators
    df_1h['EMA_50'] = df_1h['Close'].ewm(span=50, adjust=False).mean()
    df_1h['H1_Trend'] = np.where(df_1h['Close'] > df_1h['EMA_50'], 1, -1)

    df_15m['EMA_20'] = df_15m['Close'].ewm(span=20, adjust=False).mean()
    df_15m['M15_Trend'] = np.where(df_15m['Close'] > df_15m['EMA_20'], 1, -1)

    df_5m['H1_Trend'] = df_1h['H1_Trend'].reindex(df_5m.index, method='ffill')
    df_5m['M15_Trend'] = df_15m['M15_Trend'].reindex(df_5m.index, method='ffill')

    # คำนวณ ATR 14
    high_low = df_5m['High'] - df_5m['Low']
    high_cp  = np.abs(df_5m['High'] - df_5m['Close'].shift(1))
    low_cp   = np.abs(df_5m['Low'] - df_5m['Close'].shift(1))
    df_5m['ATR'] = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1).rolling(14).mean()

    # คำนวณ FVG
    df_5m['Bullish_FVG'] = df_5m['Low'] > df_5m['High'].shift(2)
    df_5m['Bearish_FVG'] = df_5m['High'] < df_5m['Low'].shift(2)

    return df_5m.dropna()

def simulate_trade(df, index, is_long, rr_ratio, sl_buffer):
    """จำลองผลลัพธ์การเทรด 1 ออเดอร์ (คืนค่า Profit/Loss Ratio)"""
    latest_bar  = df.iloc[index]
    prev_bar    = df.iloc[index-1]
    prev_2_bar  = df.iloc[index-2]

    entry_price = float(latest_bar['Close'])

    if is_long:
        sl_price  = float(min(latest_bar['Low'], prev_bar['Low'])) - sl_buffer
        risk_dist = entry_price - sl_price
        tp_price  = entry_price + (risk_dist * rr_ratio)
    else:
        sl_price  = float(max(latest_bar['High'], prev_bar['High'])) + sl_buffer
        risk_dist = sl_price - entry_price
        tp_price  = entry_price - (risk_dist * rr_ratio)

    if risk_dist <= 0:
        return None

    # Forward Simulation ไม่เกิน 100 แท่งถัดไป
    future_bars = df.iloc[index+1 : index+101]
    for _, f_bar in future_bars.iterrows():
        if is_long:
            if f_bar['High'] >= tp_price:
                return rr_ratio  # ได้กำไรตามอัตราส่วน R:R (เช่น +3.0)
            if f_bar['Low'] <= sl_price:
                return -1.0      # ขาดทุน 1 R (เช่น -1.0)
        else:
            if f_bar['Low'] <= tp_price:
                return rr_ratio
            if f_bar['High'] >= sl_price:
                return -1.0

    return 0.0  # หมดเวลาถือครอง (Breakeven/Flat)

def run_grid_search():
    """ทำการวนลูปค้นหาค่า Parameter combination ที่ให้ผลตอบแทนดีที่สุด"""
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ไม่พบไฟล์โมเดลที่ Path: {MODEL_PATH}")
        return

    model = joblib.load(MODEL_PATH)
    df = fetch_backtest_data()
    if df is None:
        return

    print("⚙️ กำลังประมวลผล Grid Search Optimization...")

    best_score = -float('inf')
    best_config = None

    # วนลูปทดสอบทุก combination
    for proba_th in PARAM_GRID['PROBA_THRESHOLD']:
        for rr in PARAM_GRID['RR_RATIO']:
            for sl_buf in PARAM_GRID['GOLD_SL_BUFFER']:
                
                total_trades = 0
                wins = 0
                total_r_return = 0.0

                for i in range(50, len(df) - 100):
                    latest_bar  = df.iloc[i]
                    prev_2_bar  = df.iloc[i-2]
                    latest_time = df.index[i]

                    is_long  = (latest_bar['H1_Trend'] == 1) and (latest_bar['M15_Trend'] == 1) and latest_bar['Bullish_FVG']
                    is_short = (latest_bar['H1_Trend'] == -1) and (latest_bar['M15_Trend'] == -1) and latest_bar['Bearish_FVG']

                    if not (is_long or is_short):
                        continue

                    entry_price = float(latest_bar['Close'])
                    if is_long:
                        sl_price  = float(min(latest_bar['Low'], df.iloc[i-1]['Low'])) - sl_buf
                        risk_dist = entry_price - sl_price
                        fvg_size  = float(latest_bar['Low'] - prev_2_bar['High'])
                    else:
                        sl_price  = float(max(latest_bar['High'], df.iloc[i-1]['High'])) + sl_buf
                        risk_dist = sl_price - entry_price
                        fvg_size  = float(prev_2_bar['Low'] - latest_bar['High'])

                    if risk_dist <= 0:
                        continue

                    # Predict ด้วย ML
                    features = pd.DataFrame([{
                        'FVG_Size': fvg_size,
                        'ATR': float(latest_bar['ATR']),
                        'Hour': int(latest_time.hour),
                        'Minute': int(latest_time.minute),
                        'Risk_Distance': risk_dist
                    }])

                    win_prob = float(model.predict_proba(features)[0][1])

                    # กรองเฉพาะสัญญาณที่ผ่าน Threshold
                    if win_prob >= proba_th:
                        trade_result = simulate_trade(df, i, is_long, rr, sl_buf)
                        if trade_result is not None:
                            total_trades += 1
                            if trade_result > 0:
                                wins += 1
                            total_r_return += trade_result

                # ประเมินคะแนน (คะแนนหลักใช้ Total R Return โดยต้องมีอย่างน้อย 5 ไม้)
                if total_trades >= 5:
                    win_rate = (wins / total_trades) * 100
                    # Score Formula: Total R Return x Win Rate Factor
                    score = total_r_return * (win_rate / 100.0)

                    if score > best_score:
                        best_score = score
                        best_config = {
                            "PROBA_THRESHOLD": round(proba_th, 2),
                            "RR_RATIO": round(rr, 2),
                            "GOLD_SL_BUFFER": round(sl_buf, 2),
                            "WIN_RATE": round(win_rate, 2),
                            "TOTAL_TRADES": total_trades,
                            "TOTAL_R_RETURN": round(total_r_return, 2),
                            "UPDATED_AT": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        }

    # บันทึกผลลัพธ์ลงไฟล์ best_config.json
    if best_config:
        print("\n" + "="*50)
        print("🎉 พบพารามิเตอร์ที่ให้ผลตอบแทนดีที่สุด (Best Configuration):")
        print(f"   • Proba Threshold : {best_config['PROBA_THRESHOLD']}")
        print(f"   • RR Ratio        : 1:{best_config['RR_RATIO']}")
        print(f"   • SL Buffer       : ${best_config['GOLD_SL_BUFFER']}")
        print(f"   • Win Rate        : {best_config['WIN_RATE']}% ({best_config['TOTAL_TRADES']} Trades)")
        print(f"   • Net R-Return    : +{best_config['TOTAL_R_RETURN']} R")
        print("="*50)

        with open(CONFIG_OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(best_config, f, indent=4, ensure_ascii=False)

        print(f"\n💾 บันทึกค่าคอนฟิกเรียบร้อยที่ไฟล์: '{CONFIG_OUTPUT_PATH}'")
    else:
        print("⚠️ ไม่พบชุดพารามิเตอร์ที่ผ่านเกณฑ์ขั้นต่ำ")

if __name__ == "__main__":
    run_grid_search()
