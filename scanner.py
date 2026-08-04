"""
Gold Real-time Scanner (scanner.py)
สคริปต์สแกนราคาทองคำ Real-time M5 SMC (Wave 3 + FVG)
ระบบสมบูรณ์แบบระดับ Production:
 1. Signal Deduplication & State Management (ป้องกันการส่งสัญญาณซ้ำ)
 2. Multi-Provider Data Fallback (yfinance -> Twelve Data -> Alpha Vantage)
 3. Economic News Filter (งดเทรดช่วงข่าว High Impact USD/XAU ก่อน-หลัง 30 นาที)
 4. Daily Rollover Guard (งดสแกนช่วง 04:00 - 05:30 น. เวลาไทย)
 5. Google Sheets Real-time Logging (บันทึกข้อมูลสำหรับ Forward Test Dashboard)
 6. ML Filtering & Dynamic Lot Scaling
 7. Risk-Free Mechanism (Break-Even Trigger at 50% TP)
"""

import os
import math
import json
import requests
import joblib
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timezone, timedelta

# Google Sheets Libraries (Optional/Graceful Fallback)
try:
    import gspread
    from oauth2client.service_account import ServiceAccountCredentials
    GSPREAD_AVAILABLE = True
except ImportError:
    GSPREAD_AVAILABLE = False

# ==================== CONFIGURATION ====================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", os.getenv("TELEGRAM_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN_HERE"))
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "YOUR_TELEGRAM_CHAT_ID_HERE")
MODEL_FILE_PATH    = os.getenv("MODEL_FILE_PATH", "gold_ml_filter.pkl")
CONFIG_FILE_PATH   = os.getenv("CONFIG_FILE_PATH", "best_config.json")
LAST_SIGNAL_PATH   = "last_signal.json"
GOOGLE_SHEET_NAME  = os.getenv("GOOGLE_SHEET_NAME", "Gold_Trading_Logs")

# API Keys สำหรับ Provider สำรอง (ใส่ใน GitHub Secrets หรือ Environment Variables)
TWELVE_DATA_API_KEY   = os.getenv("TWELVE_DATA_API_KEY", "")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")

SYMBOL              = "GC=F"
ACCOUNT_EQUITY      = float(os.getenv("ACCOUNT_EQUITY", "10000.0"))
BASE_RISK_PCT       = float(os.getenv("RISK_PCT", "0.01"))
DYNAMIC_LOT_SCALING = True

# ==================== 1. MULTI-PROVIDER DATA FETCHING ====================
def fetch_gold_data_yfinance() -> pd.DataFrame:
    """ Primary Provider: Yahoo Finance """
    print("📡 [1/3] กำลังดึงข้อมูลจาก Yahoo Finance...")
    df = yf.download(SYMBOL, period="5d", interval="5m", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()
    if len(df) < 50:
        raise ValueError("ข้อมูลจาก yfinance ไม่ครบถ้วน")
    return df

def fetch_gold_data_twelvedata() -> pd.DataFrame:
    """ Secondary Provider: Twelve Data API """
    if not TWELVE_DATA_API_KEY:
        raise ValueError("ไม่ได้ตั้งค่า TWELVE_DATA_API_KEY")
    
    print("📡 [2/3] กำลังดึงข้อมูลสำรองจาก Twelve Data API...")
    url = f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval=5min&outputsize=500&apikey={TWELVE_DATA_API_KEY}"
    res = requests.get(url, timeout=10).json()
    
    if "values" not in res:
        raise ValueError(f"Twelve Data Error: {res.get('message', 'Unknown Error')}")
    
    data = res["values"]
    df = pd.DataFrame(data)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime').sort_index()
    
    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col].astype(float)
        
    df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'})
    return df

def fetch_gold_data_alphavantage() -> pd.DataFrame:
    """ Tertiary Provider: Alpha Vantage API """
    if not ALPHA_VANTAGE_API_KEY:
        raise ValueError("ไม่ได้ตั้งค่า ALPHA_VANTAGE_API_KEY")
        
    print("📡 [3/3] กำลังดึงข้อมูลสำรองจาก Alpha Vantage API...")
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=XAUUSD&interval=5min&extended_hours=false&apikey={ALPHA_VANTAGE_API_KEY}"
    res = requests.get(url, timeout=10).json()
    
    time_series_key = "Time Series (5min)"
    if time_series_key not in res:
        raise ValueError("Alpha Vantage Rate Limit หรือ Error")
        
    data = res[time_series_key]
    df = pd.DataFrame.from_dict(data, orient='index')
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    
    df = df.rename(columns={
        '1. open': 'Open',
        '2. high': 'High',
        '3. low': 'Low',
        '4. close': 'Close'
    })
    for col in ['Open', 'High', 'Low', 'Close']:
        df[col] = df[col].astype(float)
        
    return df

def get_gold_market_data() -> pd.DataFrame:
    """ ระบบ Fallback ลำดับชั้นสำหรับดึงข้อมูลราคา """
    try:
        return fetch_gold_data_yfinance()
    except Exception as e:
        print(f"⚠️ yfinance ล้มเหลว/ถูกบล็อก: {e}")
    
    try:
        return fetch_gold_data_twelvedata()
    except Exception as e:
        print(f"⚠️ Twelve Data ล้มเหลว: {e}")

    try:
        return fetch_gold_data_alphavantage()
    except Exception as e:
        print(f"⚠️ Alpha Vantage ล้มเหลว: {e}")

    raise RuntimeError("❌ ทุก Provider ล้มเหลวในการดึงข้อมูลราคา ไม่สามารถสแกนกราฟได้")

# ==================== 2. STATE MANAGEMENT & DEDUPLICATION ====================
def is_duplicate_signal(signal_time_str: str, signal_type: str) -> bool:
    """ เช็กว่าแท่งเทียน ณ เวลา signal_time_str เคยถูกแจ้งเตือนไปแล้วหรือยัง """
    if os.path.exists(LAST_SIGNAL_PATH):
        try:
            with open(LAST_SIGNAL_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data.get("time") == signal_time_str and data.get("type") == signal_type:
                    return True
        except Exception as e:
            print(f"⚠️ ไม่สามารถอ่าน {LAST_SIGNAL_PATH}: {e}")
    return False

def save_last_signal(signal_time_str: str, signal_type: str):
    """ บันทึก State สัญญาณล่าสุดลงไฟล์ JSON """
    try:
        with open(LAST_SIGNAL_PATH, "w", encoding="utf-8") as f:
            json.dump({
                "time": signal_time_str, 
                "type": signal_type, 
                "updated_at": datetime.now().isoformat()
            }, f, ensure_ascii=False, indent=2)
        print(f"💾 บันทึก State สัญญาณสำเร็จ: {signal_time_str} ({signal_type})")
    except Exception as e:
        print(f"⚠️ ไม่สามารถบันทึก State ได้: {e}")

# ==================== 3. ROLLOVER & HIGH SPREAD GUARD ====================
def is_market_rollover_time() -> tuple[bool, str]:
    utc_now = datetime.now(timezone.utc)
    th_now = utc_now + timedelta(hours=7)
    current_time = th_now.time()

    start_time = datetime.strptime("04:00", "%H:%M").time()
    end_time   = datetime.strptime("05:30", "%H:%M").time()

    if start_time <= current_time <= end_time:
        return True, f"ช่วงตลาดปิดประจำวัน/Spread ถ่างสูง ({th_now.strftime('%H:%M')} น. BKK)"
    return False, ""

# ==================== 4. ECONOMIC NEWS FILTER ====================
def is_high_impact_news_near(buffer_minutes: int = 30) -> tuple[bool, str]:
    url = "https://nfp.ourforecast.com/api/v1/forexfactory"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            events = response.json()
            now_utc = datetime.now(timezone.utc)

            for event in events:
                currency = str(event.get("currency", "")).upper()
                impact   = str(event.get("impact", "")).capitalize()

                if currency in ["USD", "XAU"] and impact == "High":
                    event_time_str = event.get("date")
                    if event_time_str:
                        event_dt = datetime.fromisoformat(event_time_str.replace("Z", "+00:00"))
                        time_diff_minutes = (event_dt - now_utc).total_seconds() / 60.0
                        
                        if abs(time_diff_minutes) <= buffer_minutes:
                            title = event.get('title', 'High Impact Economic News')
                            status = "กำลังจะออกในอีก" if time_diff_minutes > 0 else "เพิ่งออกไปเมื่อ"
                            return True, f"{title} ({currency}) [{status} {abs(int(time_diff_minutes))} นาที]"
    except Exception as e:
        print(f"⚠️ ไม่สามารถดึงข้อมูลข่าวได้ ({e}) -> สแกนต่อตามปกติโดยไม่ใช้ News Filter")
    return False, ""

# ==================== 5. GOOGLE SHEETS LOGGING ====================
def log_signal_to_google_sheet(timestamp, signal_type, entry, sl, tp, be_trigger, win_prob):
    """ บันทึกสัญญาณการเทรดลง Google Sheets สำหรับ Forward Test """
    if not GSPREAD_AVAILABLE:
        print("ℹ️ ไม่พบกิติกรรม gspread/oauth2client -> Skip การบันทึก Google Sheet")
        return

    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds_json = os.getenv("GOOGLE_CREDENTIALS_JSON")
        
        if creds_json:
            creds_dict = json.loads(creds_json)
            creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        elif os.path.exists("google_credentials.json"):
            creds = ServiceAccountCredentials.from_json_keyfile_dict(json.load(open("google_credentials.json")), scope)
        else:
            print("ℹ️ ไม่พบ Google Credentials -> Skip การบันทึก Sheet")
            return

        client = gspread.authorize(creds)
        sheet = client.open(GOOGLE_SHEET_NAME).worksheet("Signals")

        row = [
            timestamp,
            signal_type,
            entry,
            sl,
            tp,
            be_trigger,
            f"{win_prob*100:.1f}%",
            "OPEN",  # Status
            0.0,     # Exit_Price
            0.0      # PnL
        ]
        
        sheet.append_row(row)
        print("📊 บันทึกสัญญาณลง Google Sheets เรียบร้อยแล้ว")
    except Exception as e:
        print(f"⚠️ ไม่สามารถบันทึกข้อมูลลง Google Sheet ได้: {e}")

# ==================== HELPER FUNCTIONS ====================
def load_auto_config():
    default_config = {"PROBA_THRESHOLD": 0.60, "RR_RATIO": 3.0, "GOLD_SL_BUFFER": 0.80}
    if os.path.exists(CONFIG_FILE_PATH):
        try:
            with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default_config
    return default_config

def send_telegram_alert(message: str):
    if TELEGRAM_BOT_TOKEN in ["YOUR_TELEGRAM_BOT_TOKEN_HERE", ""] or TELEGRAM_CHAT_ID in ["YOUR_TELEGRAM_CHAT_ID_HERE", ""]:
        print("ℹ️ ไม่ได้ระบุ TELEGRAM_BOT_TOKEN หรือ TELEGRAM_CHAT_ID")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        res = requests.post(url, json=payload, timeout=10).json()
        if res.get("ok"):
            print("🔔 ส่งการแจ้งเตือนไปยัง Telegram เรียบร้อยแล้ว")
        else:
            print(f"❌ ส่ง Telegram ไม่สำเร็จ: {res.get('description')}")
    except Exception as e:
        print(f"❌ ไม่สามารถเชื่อมต่อ Telegram ได้: {e}")

def calculate_lot_size(equity: float, risk_dist: float, win_prob: float) -> tuple[float, float]:
    applied_risk_pct = BASE_RISK_PCT * 1.5 if (DYNAMIC_LOT_SCALING and win_prob >= 0.65) else BASE_RISK_PCT
    risk_amount = equity * applied_risk_pct
    calculated_lot = risk_amount / (risk_dist * 100.0)
    final_lot = max(0.01, math.floor(calculated_lot * 100.0) / 100.0)
    return final_lot, applied_risk_pct * 100.0

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# ==================== MAIN SCANNER ====================
def run_scanner():
    config = load_auto_config()
    proba_threshold = float(config.get("PROBA_THRESHOLD", 0.60))
    rr_ratio        = float(config.get("RR_RATIO", 3.0))
    sl_buffer       = float(config.get("GOLD_SL_BUFFER", 0.80))

    current_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n🔍 [{current_time_str}] สแกนกราฟ M5 (Threshold: {proba_threshold*100:.0f}%)...")

    # 1. ROLLOVER GUARD
    is_rollover, rollover_msg = is_market_rollover_time()
    if is_rollover:
        print(f"🛑 [ROLLOVER GUARD] งดสแกนเนื่องจากเป็นช่วง: {rollover_msg}")
        return

    # 2. ECONOMIC NEWS FILTER
    has_news, news_title = is_high_impact_news_near(buffer_minutes=30)
    if has_news:
        print(f"🛑 [NEWS FILTER] งดสแกนสัญญาณเนื่องจากใกล้ช่วงข่าวใหญ่: {news_title}")
        return

    # 3. MODEL CHECK
    if not os.path.exists(MODEL_FILE_PATH):
        print(f"❌ ไม่พบไฟล์โมเดลที่ Path: {MODEL_FILE_PATH}")
        return

    try:
        model = joblib.load(MODEL_FILE_PATH)
        
        # 4. DATA FETCH WITH FALLBACK MECHANISM
        df_5m = get_gold_market_data()

        # Trend & Indicators
        df_15m = df_5m.resample('15min').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last'}).dropna()
        df_1h  = df_5m.resample('1H').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last'}).dropna()

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

        latest_bar  = df_5m.iloc[-2]
        prev_bar    = df_5m.iloc[-3]
        latest_time = df_5m.index[-2]
        latest_time_str = latest_time.strftime('%Y-%m-%d %H:%M')

        is_long  = (latest_bar['H1_Trend'] == 1) and (latest_bar['M15_Trend'] == 1) and latest_bar['Bullish_FVG']
        is_short = (latest_bar['H1_Trend'] == -1) and (latest_bar['M15_Trend'] == -1) and latest_bar['Bearish_FVG']

        if not (is_long or is_short):
            print("ℹ️ ไม่พบ Setup SMC ในแท่งปัจจุบัน")
            return

        signal_type = "BUY 🟢" if is_long else "SELL 🔴"

        # 5. SIGNAL DEDUPLICATION CHECK
        if is_duplicate_signal(latest_time_str, signal_type):
            print(f"ℹ️ สัญญาณ {signal_type} ของแท่ง {latest_time_str} เคยถูกส่งไปแล้ว (Skip Duplicate)")
            return

        entry_price = float(latest_bar['Close'])
        be_buffer   = 0.30
        
        if is_long:
            sl_price   = float(min(latest_bar['Low'], prev_bar['Low'])) - sl_buffer
            risk_dist  = entry_price - sl_price
            tp_price   = entry_price + (risk_dist * rr_ratio)
            be_trigger = entry_price + (risk_dist * rr_ratio * 0.5)
            be_sl      = entry_price + be_buffer
            fvg_size   = float(latest_bar['Low'] - df_5m.iloc[-4]['High'])
        else:
            sl_price   = float(max(latest_bar['High'], prev_bar['High'])) + sl_buffer
            risk_dist  = sl_price - entry_price
            tp_price   = entry_price - (risk_dist * rr_ratio)
            be_trigger = entry_price - (risk_dist * rr_ratio * 0.5)
            be_sl      = entry_price - be_buffer
            fvg_size   = float(df_5m.iloc[-4]['Low'] - latest_bar['High'])

        if risk_dist <= 0:
            return

        features = pd.DataFrame([{
            'FVG_Size': fvg_size,
            'ATR': float(latest_bar['ATR']),
            'RSI': float(latest_bar['RSI']),
            'Hour': int(latest_time.hour),
            'DayOfWeek': int(latest_time.dayofweek),
            'Risk_Distance': risk_dist
        }])

        win_prob = float(model.predict_proba(features)[0][1])
        print(f"💡 พบ Setup {signal_type} | ML Win Prob: {win_prob*100:.1f}%")

        if win_prob >= proba_threshold:
            lot_size, risk_pct_used = calculate_lot_size(ACCOUNT_EQUITY, risk_dist, win_prob)

            alert_msg = (
                f"🚨 *SIGNAL ALERT: XAU/USD ({signal_type})*\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"🤖 *ML Win Probability:* `{win_prob*100:.1f}%` (Threshold ≥ {proba_threshold*100:.0f}%)\n"
                f"📍 *Entry Price:* `${entry_price:,.2f}`\n"
                f"🛑 *Stop Loss (SL):* `${sl_price:,.2f}` (`${risk_dist:.2f}` Risk)\n"
                f"🎯 *Take Profit (TP):* `${tp_price:,.2f}` (R:R 1:{rr_ratio})\n"
                f"⚖️ *Recommended Lot:* `{lot_size}` Lot (Risk {risk_pct_used:.1f}%)\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"🛡️ *RISK-FREE MECHANISM (Break-Even):*\n"
                f" 🔹 *BE Trigger (50% TP):* `${be_trigger:,.2f}`\n"
                f" 🔹 *Action:* ถ้าราคาถึง BE Trigger ให้เลื่อน SL มาที่ `${be_sl:,.2f}` (ล็อกหน้าไม้)\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"⏰ *Time:* `{latest_time_str}`\n"
                f"💡 _ยกระดับด้วย SMC + ML Filter + DevOps Protection_"
            )

            print(alert_msg)
            send_telegram_alert(alert_msg)
            save_last_signal(latest_time_str, signal_type)
            log_signal_to_google_sheet(latest_time_str, signal_type, entry_price, sl_price, tp_price, be_trigger, win_prob)
        else:
            print(f"⛔ สัญญาณถูกปฏิเสธเนื่องจากความมั่นใจ ML ({win_prob*100:.1f}%) ต่ำกว่าเกณฑ์")

    except Exception as e:
        print(f"⚠️ เกิดข้อผิดพลาดขณะสแกนตลาด: {e}")

if __name__ == "__main__":
    run_scanner()
 
