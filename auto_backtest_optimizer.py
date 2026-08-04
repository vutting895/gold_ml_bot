"""
Auto Backtest Optimizer (auto_backtest_optimizer.py)
สคริปต์ค้นหาค่า Config ที่ดีที่สุด (Grid Search) ผ่านการทำ Backtest
"""

import os
import json
import joblib
import itertools
import numpy as np
import pandas as pd
import yfinance as yf

MODEL_FILE_PATH  = os.getenv("MODEL_FILE_PATH", "gold_ml_filter.pkl")
CONFIG_FILE_PATH = os.getenv("CONFIG_FILE_PATH", "best_config.json")

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def prepare_backtest_data():
    print("📥 กำลังดึงข้อมูลราคาทองคำ (GC=F) สำหรับ Backtest ย้อนหลัง 60d...")
    df_5m = yf.download("GC=F", period="60d", interval="5m", progress=False)
    if isinstance(df_5m.columns, pd.MultiIndex):
        df_5m.columns = df_5m.columns.get_level_values(0)
    
    df_5m = df_5m.dropna()

    # แก้ไข '1H' เป็น '1h' เพื่อป้องกัน FutureWarning
    df_15m = df_5m.resample('15min').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last'}).dropna()
    df_1h  = df_5m.resample('1h').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last'}).dropna()

    df_1h['EMA_50'] = df_1h['Close'].ewm(span=50, adjust=False).mean()
    df_1h['H1_Trend'] = np.where(df_1h['Close'] > df_1h['EMA_50'], 1, -1)

    df_15m['EMA_20'] = df_15m['Close'].ewm(span=20, adjust=False).mean()
    df_15m['M15_Trend'] = np.where(df_15m['Close'] > df_15m['EMA_20'], 1, -1)

    df_5m['H1_Trend'] = df_1h['H1_Trend'].reindex(df_5m.index, method='ffill')
    df_5m['M15_Trend'] = df_15m['M15_Trend'].reindex(df_5m.index, method='ffill')

    high_low = df_5m['High'] - df_5m['Low']
    high_cp  = np.abs(df_5m['High'] - df_5m['Close'].shift(1))
    low_cp   = np.abs(df_5m['Low'] - df_5m['Close'].shift(1))
    df_5m['ATR'] = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1).rolling(14).mean()
    df_5m['RSI'] = calculate_rsi(df_5m['Close'], 14)

    df_5m['Bullish_FVG'] = df_5m['Low'] > df_5m['High'].shift(2)
    df_5m['Bearish_FVG'] = df_5m['High'] < df_5m['Low'].shift(2)

    return df_5m

def run_grid_search():
    if not os.path.exists(MODEL_FILE_PATH):
        print(f"❌ ไม่พบไฟล์โมเดล {MODEL_FILE_PATH}")
        return

    model = joblib.load(MODEL_FILE_PATH)
    df = prepare_backtest_data()

    param_grid = {
        'PROBA_THRESHOLD': [0.55, 0.60, 0.65, 0.70],
        'RR_RATIO': [2.0, 2.5, 3.0],
        'GOLD_SL_BUFFER': [0.50, 0.80, 1.00]
    }

    grid = list(itertools.product(
        param_grid['PROBA_THRESHOLD'],
        param_grid['RR_RATIO'],
        param_grid['GOLD_SL_BUFFER']
    ))

    print("⚙️ กำลังประมวลผล Grid Search Optimization...")
    best_score = -float('inf')
    best_config = {}

    for threshold, rr, sl_buf in grid:
        total_pnl = 0.0
        win_count = 0
        loss_count = 0

        for i in range(4, len(df) - 50):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            time_idx = df.index[i]

            is_long  = (row['H1_Trend'] == 1) and (row['M15_Trend'] == 1) and row['Bullish_FVG']
            is_short = (row['H1_Trend'] == -1) and (row['M15_Trend'] == -1) and row['Bearish_FVG']

            if not (is_long or is_short):
                continue

            entry_price = float(row['Close'])
            
            if is_long:
                sl_price  = float(min(row['Low'], prev_row['Low'])) - sl_buf
                risk_dist = entry_price - sl_price
                fvg_size  = float(row['Low'] - df.iloc[i-3]['High'])
            else:
                sl_price  = float(max(row['High'], prev_row['High'])) + sl_buf
                risk_dist = sl_price - entry_price
                fvg_size  = float(df.iloc[i-3]['Low'] - row['High'])

            if risk_dist <= 0:
                continue

            # จัดเรียงคอลัมน์ Features ให้ตรงกับตอน Train โมเดล
            features = pd.DataFrame([{
                'FVG_Size': fvg_size,
                'ATR': float(row['ATR']),
                'RSI': float(row['RSI']),
                'Hour': int(time_idx.hour),
                'DayOfWeek': int(time_idx.dayofweek),
                'Risk_Distance': risk_dist
            }])

            win_prob = float(model.predict_proba(features)[0][1])

            if win_prob >= threshold:
                tp_price = entry_price + (risk_dist * rr) if is_long else entry_price - (risk_dist * rr)
                future_prices = df.iloc[i+1:i+50]

                trade_won = False
                for _, f_row in future_prices.iterrows():
                    if is_long:
                        if f_row['High'] >= tp_price:
                            trade_won = True
                            break
                        elif f_row['Low'] <= sl_price:
                            break
                    else:
                        if f_row['Low'] <= tp_price:
                            trade_won = True
                            break
                        elif f_row['High'] >= sl_price:
                            break

                if trade_won:
                    total_pnl += (risk_dist * rr)
                    win_count += 1
                else:
                    total_pnl -= risk_dist
                    loss_count += 1

        total_trades = win_count + loss_count
        if total_trades > 0 and total_pnl > best_score:
            best_score = total_pnl
            best_config = {
                "PROBA_THRESHOLD": threshold,
                "RR_RATIO": rr,
                "GOLD_SL_BUFFER": sl_buf,
                "BACKTEST_WIN_RATE": round((win_count / total_trades) * 100, 2),
                "TOTAL_TRADES": total_trades,
                "ESTIMATED_NET_PNL": round(total_pnl, 2)
            }

    if best_config:
        print(f"🎯 ค้นพบ Config ที่ดีที่สุด: {best_config}")
        with open(CONFIG_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(best_config, f, indent=2, ensure_ascii=False)
        print(f"💾 บันทึกค่าลงใน {CONFIG_FILE_PATH} เรียบร้อยแล้ว")
    else:
        print("⚠️ ไม่พบ Config ที่ทำกำไรได้ในรอบ Backtest นี้")

if __name__ == "__main__":
    run_grid_search()
    
