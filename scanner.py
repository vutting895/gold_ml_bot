import json
import os
from datetime import datetime, timedelta
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
# 1. TECHNICAL INDICATORS & MTF ANALYSIS
# ==========================================
def add_indicators(df):
    """คำนวณ ATR (14) และ RSI (14) ตามมาตรฐานสากล"""
    df = df.copy()

    # 1. True Range & ATR (14)
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    # 2. RSI (14)
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    return df


def get_h1_trend_filter():
    """Multi-Timeframe Analysis: ดึงกราฟ H1 เพื่อเช็กแนวโน้มหลัก (EMA 20 vs EMA 50)"""
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": "XAU/USD",
        "interval": "1h",
        "outputsize": 60,
        "apikey": TWELVE_DATA_API_KEY,
        "format": "JSON",
        "timezone": "Asia/Bangkok",
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        if "values" not in data:
            print("⚠️ [MTF Filter] ไม่สามารถดึงกราฟ H1 ได้ ใช้การสแกนแบบไม่กรอง MTF")
            return "BOTH"

        parsed_data = []
        for c in reversed(data["values"]):
            parsed_data.append({"close": float(c["close"])})
        df_h1 = pd.DataFrame(parsed_data)

        # คำนวณ EMA 20 และ EMA 50 บน Timeframe H1
        df_h1["ema20"] = df_h1["close"].ewm(span=20, adjust=False).mean()
        df_h1["ema50"] = df_h1["close"].ewm(span=50, adjust=False).mean()

        latest = df_h1.iloc[-1]
        if latest["ema20"] > latest["ema50"]:
            print("📈 [MTF Filter] H1 Trend = BULLISH (อนุญาตเฉพาะ BUY)")
            return "BUY_ONLY"
        elif latest["ema20"] < latest["ema50"]:
            print("📉 [MTF Filter] H1 Trend = BEARISH (อนุญาตเฉพาะ SELL)")
            return "SELL_ONLY"
        return "BOTH"

    except Exception as e:
        print(f"⚠️ [MTF Filter Error]: {e}")
        return "BOTH"


# ==========================================
# 2. DEMAND & SUPPLY ZONES DETECTOR
# ==========================================
def detect_demand_supply_zones(df, window=20):
    """คำนวณหา Demand Zone (โซนรับ) และ Supply Zone (โซนต้าน)"""
    if len(df) < window:
        return {"demand": None, "supply": None}

    recent_df = df.iloc[-window:]

    # Supply Zone: บริเวณสูงสุดของ Swing High ย้อนหลัง
    supply_high = recent_df["high"].max()
    supply_low = supply_high - (recent_df["atr"].iloc[-1] * 0.5 if "atr" in recent_df.columns else 1.0)

    # Demand Zone: บริเวณต่ำสุดของ Swing Low ย้อนหลัง
    demand_low = recent_df["low"].min()
    demand_high = demand_low + (recent_df["atr"].iloc[-1] * 0.5 if "atr" in recent_df.columns else 1.0)

    return {
        "demand": (demand_low, demand_high),
        "supply": (supply_low, supply_high),
    }


def is_price_near_zone(price, zone):
    """เช็กว่าราคาปัจจุบันอยู่ในหรือใกล้เคียงกับ Demand/Supply Zone หรือไม่"""
    if not zone:
        return False
    low_bound, high_bound = zone
    return (low_bound - 0.5) <= price <= (high_bound + 0.5)


# ==========================================
# 3. SNIPER SCORE ENGINE
# ==========================================
def calculate_sniper_score(signal_type, price, fvg_size, atr, rsi, mtf_permission, zones):
    """คำนวณ Sniper Score (0-100) ประเมินความน่าจะเป็นและคุณภาพสัญญาณ"""
    score = 40  # Base Score

    # 1. เช็กความสอดคล้องกับ H1 MTF Trend (+20 คะแนน)
    if (signal_type == "BUY" and mtf_permission == "BUY_ONLY") or (signal_type == "SELL" and mtf_permission == "SELL_ONLY"):
        score += 20
    elif mtf_permission == "BOTH":
        score += 10

    # 2. เช็กความสอดคล้องกับ Demand / Supply Zone (+20 คะแนน)
    demand_zone = zones.get("demand")
    supply_zone = zones.get("supply")

    if signal_type == "BUY" and is_price_near_zone(price, demand_zone):
        score += 20
        zone_info = "Demand Zone 🟢"
    elif signal_type == "SELL" and is_price_near_zone(price, supply_zone):
        score += 20
        zone_info = "Supply Zone 🔴"
    else:
        zone_info = "Mid-Range / Breakout ⚪"

    # 3. เช็กขนาด FVG Gap เทียบกับ ATR Momentum (+10 คะแนน)
    if fvg_size >= (1.2 * atr):
        score += 10
    elif fvg_size >= (0.8 * atr):
        score += 5

    # 4. เช็กค่า RSI Overbought / Oversold Confirmation (+10 คะแนน)
    if signal_type == "BUY" and rsi < 50:
        score += 10
    elif signal_type == "SELL" and rsi > 50:
        score += 10

    return min(score, 100), zone_info


# ==========================================
# 4. HIGH-IMPACT NEWS FILTER
# ==========================================
def is_high_impact_news_near(window_minutes=30):
    """ตรวจสอบข่าวแรงเกี่ยวกับ USD (High Impact) ก่อนและหลังข่าวออก 30 นาที"""
    try:
        url = "https://nws.forexfactory.com/news/get_news_json.php"
        response = requests.get(url, timeout=5)
        if response.status_code != 200:
            return False, ""

        news_data = response.json()
        now_utc = datetime.now(pytz.utc)

        for news in news_data:
            if news.get("country") == "USD" and news.get("impact") == "High":
                news_time_str = news.get("date")
                if news_time_str:
                    news_dt = pd.to_datetime(news_time_str).tz_convert(pytz.utc)
                    diff_minutes = abs((news_dt - now_utc).total_seconds()) / 60.0

                    if diff_minutes <= window_minutes:
                        title = news.get("title", "USD High Impact News")
                        return True, title
        return False, ""
    except Exception:
        return False, ""


# ==========================================
# 5. DYNAMIC POSITION SIZING CALCULATOR
# ==========================================
def calculate_lot_size(balance=10000.0, risk_percent=1.0, entry=0.0, sl=0.0):
    """คำนวณขนาด Lot Size อัตโนมัติสำหรับ XAU/USD ตามหลัก Money Management"""
    try:
        risk_amount = balance * (risk_percent / 100.0)
        risk_distance = abs(entry - sl)
        if risk_distance <= 0:
            return 0.01

        lot_size = risk_amount / (risk_distance * 100.0)
        return max(round(lot_size, 2), 0.01)
    except Exception:
        return 0.01


# ==========================================
# 6. TELEGRAM SYSTEM & COMMAND HANDLER
# ==========================================
def send_telegram_message(message: str) -> bool:
    """ส่งข้อความไปยัง Telegram Bot ผ่าน HTTP POST"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("❌ [Telegram Error] ไม่พบ Token หรือ Chat ID")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    try:
        res = requests.post(url, json=payload, timeout=10)
        res.raise_for_status()
        print("✅ [Telegram] ส่งข้อความแจ้งเตือนสำเร็จ")
        return True
    except Exception as e:
        print(f"❌ [Telegram Error]: {e}")
        return False


def send_signal_alert(symbol, signal_type, entry, sl, tp, timeframe="M5", fvg_size=None, sniper_score=0, zone_info=""):
    """การ์ดแจ้งเตือนสัญญาณเทรด"""
    tz_th = pytz.timezone("Asia/Bangkok")
    now_th = datetime.now(tz_th).strftime("%Y-%m-%d %H:%M:%S")

    is_buy = signal_type.upper() == "BUY"
    type_emoji = "🟢 BUY" if is_buy else "🔴 SELL"
    trend_icon = "📈" if is_buy else "📉"

    risk_pips = abs(entry - sl)
    reward_pips = abs(tp - entry)
    rr_ratio = round(reward_pips / risk_pips, 2) if risk_pips > 0 else 0.0

    recommended_lot = calculate_lot_size(balance=10000.0, risk_percent=1.0, entry=entry, sl=sl)
    score_badge = "🔥 A+ EXCELLENT" if sniper_score >= 80 else "✅ A GOOD"

    message = (
        f"🎯 <b>{symbol} {timeframe} SNIPER SIGNAL ({score_badge})</b> {trend_icon}\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"<b>Type:</b> {type_emoji}\n"
        f"<b>Sniper Score:</b> <code>{sniper_score}/100</code> ⭐️\n"
        f"<b>Market Zone:</b> <code>{zone_info}</code>\n"
        f"<b>Time (TH):</b> <code>{now_th}</code> (UTC+7)\n\n"
        f"🎯 <b>Entry Price:</b> <code>${entry:.2f}</code>\n"
        f"🛑 <b>Stop Loss (SL):</b> <code>${sl:.2f}</code> (Risk: ${risk_pips:.2f})\n"
        f"🏁 <b>Take Profit (TP):</b> <code>${tp:.2f}</code> (Reward: ${reward_pips:.2f})\n"
        f"⚖️ <b>Risk : Reward:</b> <code>1:{rr_ratio}</code>\n"
        f"💼 <b>Recommended Lot (Risk 1% / $10k):</b> <code>{recommended_lot} Lot</code>\n"
    )

    if fvg_size:
        message += f"📐 <b>FVG Gap Size:</b> <code>${fvg_size:.2f}</code>\n"

    message += (
        "━━━━━━━━━━━━━━━━━━\n"
        "🤖 <i>Validated by Voom Sniper Engine, News Filter, MTF & ML</i>\n"
    )

    return send_telegram_message(message)


def process_telegram_commands(sheet):
    """Interactive Telegram Bot: อ่านคำสั่งจากผู้ใช้ (/status, /help)"""
    if not TELEGRAM_BOT_TOKEN:
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
    try:
        res = requests.get(url, timeout=5)
        if res.status_code != 200:
            return

        updates = res.json().get("result", [])
        for update in updates:
            message_data = update.get("message", {})
            text = message_data.get("text", "").strip()
            chat_id = message_data.get("chat", {}).get("id")

            if str(chat_id) != str(TELEGRAM_CHAT_ID) or not text.startswith("/"):
                continue

            if text == "/status":
                if sheet:
                    records = sheet.get_all_records()
                    df_rec = pd.DataFrame(records)
                    if not df_rec.empty and "Status" in df_rec.columns:
                        wins = len(df_rec[df_rec["Status"].str.upper() == "WIN"])
                        losses = len(df_rec[df_rec["Status"].str.upper() == "LOSS"])
                        opens = len(df_rec[df_rec["Status"].str.upper() == "OPEN"])
                        total = wins + losses
                        wr = (wins / total * 100) if total > 0 else 0.0

                        pnl = pd.to_numeric(df_rec["PnL"], errors="coerce").fillna(0.0).sum()

                        msg = (
                            f"📊 <b>BOT STATUS REPORT</b>\n"
                            f"━━━━━━━━━━━━━━━━━━\n"
                            f"🟢 Win: <b>{wins}</b> | 🔴 Loss: <b>{losses}</b> | ⏳ Open: <b>{opens}</b>\n"
                            f"🎯 Win Rate: <b>{wr:.1f}%</b>\n"
                            f"💰 Total PnL: <b>${pnl:+.2f}</b>"
                        )
                    else:
                        msg = "ℹ️ ยังไม่มีประวัติการเทรดในระบบ"
                else:
                    msg = "❌ ไม่สามารถเชื่อมต่อ Google Sheets ได้"
                send_telegram_message(msg)

            elif text == "/help":
                msg = (
                    "🤖 <b>COMMAND MENU</b>\n"
                    "━━━━━━━━━━━━━━━━━━\n"
                    "/status - ตรวจสอบสรุปผล Win Rate และ PnL\n"
                    "/scan - สั่งรันสแกนตลาดทองคำทันที\n"
                    "/help - แสดงเมนูช่วยเหลือ"
                )
                send_telegram_message(msg)

    except Exception as e:
        print(f"⚠️ [Telegram Command Handler Error]: {e}")


# ==========================================
# 7. DATA FETCHING & GOOGLE SHEETS
# ==========================================
def fetch_twelvedata_m5_data(count=100):
    """ดึงข้อมูลราคาย้อนหลัง M5 XAU/USD จาก Twelve Data API"""
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
            df = add_indicators(df)
        return df
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการดึงข้อมูลราคา: {e}")
        return pd.DataFrame()


def get_google_sheet_handle():
    """เชื่อมต่อ Google Sheets"""
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
    """ดึงเวลาสัญญาณล่าสุด"""
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
    """อัปเดตออเดอร์สถานะ OPEN ใน Google Sheets ว่าชน TP หรือ SL หรือยัง"""
    if not sheet or df_prices.empty:
        return

    try:
        records = sheet.get_all_records()
        if not records:
            return

        print("🔍 กำลังตรวจสอบออเดอร์ที่เปิดค้างไว้ (OPEN)...")
        tz_th = pytz.timezone("Asia/Bangkok")

        for idx, row in enumerate(records, start=2):
            if str(row.get("Status", "")).upper() == "OPEN":
                entry = float(row["Entry"])
                sl = float(row["SL"])
                tp = float(row["TP"])
                trade_type = str(row["Type"]).upper()

                signal_time = pd.to_datetime(row["Time (UTC+7)"])
                if signal_time.tzinfo is None:
                    signal_time = tz_th.localize(signal_time)

                df_index_tz = (
                    df_prices.index.tz_localize(tz_th)
                    if df_prices.index.tzinfo is None
                    else df_prices.index
                )
                future_candles = df_prices[df_index_tz > signal_time]

                for _, candle in future_candles.iterrows():
                    high = candle["high"]
                    low = candle["low"]

                    if trade_type == "BUY":
                        if low <= sl:
                            pnl = round(sl - entry, 2)
                            sheet.update_cell(idx, 6, "LOSS")
                            sheet.update_cell(idx, 7, pnl)
                            print(f"❌ Order BUY เวลา {row['Time (UTC+7)']} ชน SL (PnL: ${pnl})")
                            break
                        elif high >= tp:
                            pnl = round(tp - entry, 2)
                            sheet.update_cell(idx, 6, "WIN")
                            sheet.update_cell(idx, 7, pnl)
                            print(f"🎯 Order BUY เวลา {row['Time (UTC+7)']} ชน TP (PnL: ${pnl})")
                            break

                    elif trade_type == "SELL":
                        if high >= sl:
                            pnl = round(entry - sl, 2)
                            sheet.update_cell(idx, 6, "LOSS")
                            sheet.update_cell(idx, 7, pnl)
                            print(f"❌ Order SELL เวลา {row['Time (UTC+7)']} ชน SL (PnL: ${pnl})")
                            break
                        elif low <= tp:
                            pnl = round(entry - tp, 2)
                            sheet.update_cell(idx, 6, "WIN")
                            sheet.update_cell(idx, 7, pnl)
                            print(f"🎯 Order SELL เวลา {row['Time (UTC+7)']} ชน TP (PnL: ${pnl})")
                            break

    except Exception as e:
        print(f"⚠️ เกิดข้อผิดพลาดในการอัปเดตสถานะออเดอร์: {e}")


# ==========================================
# 8. SMC DETECTOR & MAIN SCANNER LOGIC
# ==========================================
def detect_smc_fvg(df):
    """ตรวจจับ Fair Value Gap (FVG) ร่วมกับ Dynamic ATR SL"""
    if len(df) < 3:
        return None

    i = len(df) - 2
    if i < 2:
        return None

    c1 = df.iloc[i - 2]
    c2 = df.iloc[i - 1]
    c3 = df.iloc[i]

    time_str = df.index[i].strftime("%Y-%m-%d %H:%M:%S")

    latest_atr = df["atr"].iloc[i] if "atr" in df.columns and not pd.isna(df["atr"].iloc[i]) else 1.5
    sl_buffer = max(0.5 * latest_atr, 1.0)

    # Bullish FVG
    if c3["low"] > c1["high"]:
        fvg_size = c3["low"] - c1["high"]
        return {
            "type": "BUY",
            "time": time_str,
            "entry": c3["close"],
            "fvg_size": fvg_size,
            "sl": c1["low"] - sl_buffer,
            "atr": latest_atr,
            "rsi": df["rsi"].iloc[i] if "rsi" in df.columns else 50.0,
        }

    # Bearish FVG
    elif c3["high"] < c1["low"]:
        fvg_size = c1["low"] - c3["high"]
        return {
            "type": "SELL",
            "time": time_str,
            "entry": c3["close"],
            "fvg_size": fvg_size,
            "sl": c1["high"] + sl_buffer,
            "atr": latest_atr,
            "rsi": df["rsi"].iloc[i] if "rsi" in df.columns else 50.0,
        }

    return None


def main():
    print("🚀 เริ่มต้นระบบ Gold SMC Full-Featured Scanner with Voom Sniper Engine (UTC+7)...")

    # 1. เช็กวันเสาร์-อาทิตย์ (ตลาดปิด)
    tz_th = pytz.timezone("Asia/Bangkok")
    now_th = datetime.now(tz_th)
    if now_th.weekday() in [5, 6]:
        print("😴 ตลาดทองคำปิดทำการ (วันเสาร์-อาทิตย์) ข้ามการสแกน")
        return

    sheet = get_google_sheet_handle()

    # 2. อ่านและตอบรับคำสั่งผู้ใช้ผ่าน Telegram (/status, /help)
    process_telegram_commands(sheet)

    # 3. เช็ก News Filter (ข่าวแรง USD)
    is_news, news_title = is_high_impact_news_near(window_minutes=30)
    if is_news:
        print(f"⚠️ [News Filter] งดส่งสัญญาณเนื่องจากใกล้ช่วงข่าวแรง
