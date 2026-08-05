import json
import os
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
import streamlit as st

# ตั้งค่าหน้าจอ Streamlit
st.set_page_config(
    page_title="Gold SMC Trading Dashboard (XAU/USD)",
    page_icon="🪙",
    layout="wide",
)

# โหลดค่าตัวแปรสภาพแวดล้อม ( Environment Variables / Streamlit Secrets )
GOOGLE_CREDENTIALS_JSON = os.environ.get(
    "GOOGLE_CREDENTIALS_JSON"
) or st.secrets.get("GOOGLE_CREDENTIALS_JSON")
GOOGLE_SHEET_NAME = (
    os.environ.get("GOOGLE_SHEET_NAME")
    or st.secrets.get("GOOGLE_SHEET_NAME")
    or "Gold_Trading_Logs"
)


# ดึงข้อมูลจาก Google Sheets
@st.cache_data(ttl=10)  # Cache ข้อมูล 10 วินาทีเพื่อลดการดึงข้อมูลถี่เกินไป
def load_signals():
  if not GOOGLE_CREDENTIALS_JSON:
    st.error("❌ ไม่พบข้อมูล GOOGLE_CREDENTIALS_JSON ใน Secrets")
    return pd.DataFrame()

  try:
    creds_dict = json.loads(GOOGLE_CREDENTIALS_JSON)
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
    client = gspread.authorize(creds)

    spreadsheet = client.open(GOOGLE_SHEET_NAME)
    sheet = spreadsheet.worksheet("Signals")

    data = sheet.get_all_values()
    if len(data) > 1:
      headers = data[0]
      df = pd.DataFrame(data[1:], columns=headers)
      df = df.loc[:, df.columns != ""]
      return df
    else:
      return pd.DataFrame()

  except Exception as e:
    st.error(f"เกิดข้อผิดพลาดในการเชื่อมต่อ Google Sheets: {e}")
    return pd.DataFrame()


# --- HEADER ---
st.title("🪙 Gold SMC Real-Time Dashboard (XAU/USD M5)")
st.caption(
    "ระบบตรวจจับสัญญาณ Fair Value Gap (FVG) คู่ทองคำ"
    " และวิเคราะห์ผลการเทรดอัตโนมัติ"
)

# --- SIDEBAR & MANUAL TRIGGER ---
st.sidebar.header("⚙️ เมนูและการตั้งค่า")

if st.sidebar.button("🔄 รีเฟรชข้อมูล", use_container_width=True):
  st.cache_data.clear()
  st.rerun()

st.sidebar.divider()
st.sidebar.subheader("🎯 สั่งรัน Scanner ทองคำ")
if st.sidebar.button("🚀 สแกนตลาดทองคำ (XAU/USD)", use_container_width=True):
  with st.spinner("กำลังดึงข้อมูลกราฟ XAU/USD M5 และสแกนตลาด..."):
    try:
      import scanner

      scanner.main()
      st.cache_data.clear()
      st.sidebar.success("สแกนทองคำเรียบร้อยแล้ว!")
      st.rerun()
    except Exception as e:
      st.sidebar.error(f"เกิดข้อผิดพลาดในการสแกน: {e}")

# โหลดข้อมูล
df_signals = load_signals()

# --- METRIC CARDS ---
if not df_signals.empty and "Status" in df_signals.columns:
  total_signals = len(df_signals)

  # คำนวณสถานะ Win / Loss / Open
  status_series = df_signals["Status"].astype(str).str.upper()
  win_count = len(df_signals[status_series == "WIN"])
  loss_count = len(df_signals[status_series == "LOSS"])
  open_count = len(df_signals[status_series == "OPEN"])

  closed_trades = win_count + loss_count
  win_rate = (win_count / closed_trades * 100) if closed_trades > 0 else 0.0

  # คำนวณ PnL
  if "PnL" in df_signals.columns:
    pnl_numeric = pd.to_numeric(df_signals["PnL"], errors="coerce").fillna(0.0)
    total_pnl = pnl_numeric.sum()
  else:
    total_pnl = 0.0

  # คำนวณ BUY / SELL
  buy_count = (
      len(df_signals[df_signals["Type"].astype(str).str.upper() == "BUY"])
      if "Type" in df_signals.columns
      else 0
  )
  sell_count = (
      len(df_signals[df_signals["Type"].astype(str).str.upper() == "SELL"])
      if "Type" in df_signals.columns
      else 0
  )

  time_col = (
      "Time (UTC+7)"
      if "Time (UTC+7)" in df_signals.columns
      else df_signals.columns[0]
  )
  last_time = df_signals.iloc[-1].get(time_col, "N/A")

  # แถวที่ 1: ผลประกอบการหลัก
  col1, col2, col3, col4 = st.columns(4)
  col1.metric("สัญญาณทั้งหมด", f"{total_signals} สัญญาณ")
  col2.metric(
      "Win Rate (%)",
      f"{win_rate:.1f}%",
      help="คำนวณจากออเดอร์ที่ปิดแล้ว: WIN / (WIN + LOSS)",
  )
  col3.metric(
      "ผลการเทรด (Win / Loss / Open)",
      f"🟢 {win_count} | 🔴 {loss_count} | ⏳ {open_count}",
  )
  col4.metric("รวม PnL ทั้งหมด ($)", f"${total_pnl:+.2f}")

  st.markdown("---")

  # แถวที่ 2: ข้อมูลแยกรายละเอียด
  col5, col6, col7, col8 = st.columns(4)
  col5.metric("BUY Signals", f"{buy_count}")
  col6.metric("SELL Signals", f"{sell_count}")
  col7.metric("ออเดอร์ที่ปิดแล้ว", f"{closed_trades} ออเดอร์")
  col8.metric("สัญญาณล่าสุด", f"{last_time}")
else:
  col1, col2, col3, col4 = st.columns(4)
  col1.metric("สัญญาณทั้งหมด", "0")
  col2.metric("Win Rate (%)", "0.0%")
  col3.metric("ผลการเทรด (Win / Loss / Open)", "🟢 0 | 🔴 0 | ⏳ 0")
  col4.metric("รวม PnL ทั้งหมด ($)", "$0.00")

st.divider()

# --- TABS FOR DATA DISPLAY ---
tab1, tab2 = st.tabs(["📊 รายการสัญญาณเทรดทองคำ", "⚙️ ค่า Config ปัจจุบัน"])

with tab1:
  st.subheader("📋 ประวัติสัญญาณ XAU/USD ทั้งหมด")

  if not df_signals.empty:
    df_display = df_signals.iloc[::-1].reset_index(drop=True)

    col_f1, col_f2 = st.columns(2)
    with col_f1:
      if "Type" in df_display.columns:
        filter_type = st.selectbox(
            "กรองตามประเภทสัญญาณ:", ["ทั้งหมด", "BUY", "SELL"]
        )
        if filter_type != "ทั้งหมด":
          df_display = df_display[
              df_display["Type"].astype(str).str.upper() == filter_type
          ]
    with col_f2:
      if "Status" in df_display.columns:
        filter_status = st.selectbox(
            "กรองตามสถานะออเดอร์:", ["ทั้งหมด", "WIN", "LOSS", "OPEN"]
        )
        if filter_status != "ทั้งหมด":
          df_display = df_display[
              df_display["Status"].astype(str).str.upper() == filter_status
          ]

    numeric_cols = ["Entry", "SL", "TP", "PnL"]
    for col in numeric_cols:
      if col in df_display.columns:
        df_display[col] = pd.to_numeric(df_display[col], errors="coerce")

    st.dataframe(df_display, use_container_width=True, hide_index=True)
  else:
    st.info("ยังไม่มีข้อมูลสัญญาณทองคำในระบบ")

with tab2:
  st.subheader("🛠️ การตั้งค่าระบบ (config.json)")
  config_file = "config.json"
  if os.path.exists(config_file):
    try:
      with open(config_file, "r", encoding="utf-8") as f:
        config_data = json.load(f)
      st.json(config_data)
    except Exception as e:
      st.error(f"อ่านไฟล์ config.json ผิดพลาด: {e}")
  else:
    st.info("ไม่พบไฟล์ config.json (ใช้ค่า Default)")
    st.json({
        "symbol": "XAU/USD",
        "timeframe": "5m",
        "rr_ratio": 2.0,
        "min_fvg_size": 0.5,
        "sl_buffer": "Dynamic ATR (0.5 * ATR)",
        "status": "default",
    })
      
