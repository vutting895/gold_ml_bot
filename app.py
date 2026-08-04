import json
import os
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
import streamlit as st

# ตั้งค่าหน้าจอ Streamlit
st.set_page_config(
    page_title="Gold SMC Trading Dashboard", page_icon="🪙", layout="wide"
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
    st.error("❌ ไม่พบข้อมูล GOOGLE_CREDENTIALS_JSON")
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
    data = sheet.get_all_records()
    df = pd.DataFrame(data)
    return df
  except Exception as e:
    st.error(f"เกิดข้อผิดพลาดในการเชื่อมต่อ Google Sheets: {e}")
    return pd.DataFrame()


# --- HEADER ---
st.title("🪙 Gold SMC Real-Time Dashboard (UTC+7)")
st.caption("ระบบตรวจจับสัญญาณ Fair Value Gap (FVG) และเก็บบันทึกอัตโนมัติ")

# --- SIDEBAR & MANUAL TRIGGER ---
st.sidebar.header("⚙️ เมนูและการตั้งค่า")

if st.sidebar.button("🔄 รีเฟรชข้อมูล", use_container_width=True):
  st.cache_data.clear()
  st.rerun()

st.sidebar.divider()
st.sidebar.subheader("🎯 สั่งรัน Scanner")
if st.sidebar.button("🚀 สแกนตลาดทันที", use_container_width=True):
  with st.spinner("กำลังดึงข้อมูลและสแกนตลาด..."):
    try:
      import scanner

      scanner.main()
      st.cache_data.clear()
      st.sidebar.success("สแกนเรียบร้อยแล้ว!")
      st.rerun()
    except Exception as e:
      st.sidebar.error(f"เกิดข้อผิดพลาดในการสแกน: {e}")

# โหลดข้อมูล
df_signals = load_signals()

# --- METRIC CARDS ---
col1, col2, col3, col4 = st.columns(4)

if not df_signals.empty:
  total_signals = len(df_signals)
  buy_count = len(df_signals[df_signals["Type"] == "BUY"])
  sell_count = len(df_signals[df_signals["Type"] == "SELL"])
  last_time = df_signals.iloc[-1].get("Time (UTC+7)", "N/A")

  col1.metric("สัญญาณทั้งหมด", f"{total_signals} สัญญาณ")
  col2.metric("BUY Signals", f"{buy_count}", delta_color="normal")
  col3.metric("SELL Signals", f"{sell_count}", delta_color="inverse")
  col4.metric("สัญญาณล่าสุด", f"{last_time}")
else:
  col1.metric("สัญญาณทั้งหมด", "0")
  col2.metric("BUY Signals", "0")
  col3.metric("SELL Signals", "0")
  col4.metric("สัญญาณล่าสุด", "-")

st.divider()

# --- TABS FOR DATA DISPLAY ---
tab1, tab2 = st.tabs(["📊 รายการสัญญาณเทรด", "⚙️ ค่า Config ปัจจุบัน"])

with tab1:
  st.subheader("📋 ประวัติสัญญาณทั้งหมด")

  if not df_signals.empty:
    # เรียงให้สัญญาณใหม่อยู่บนสุด
    df_display = df_signals.iloc[::-1].reset_index(drop=True)

    # ตัวกรองเลือกดูตามประเภท BUY / SELL
    filter_type = st.selectbox("กรองตามประเภท:", ["ทั้งหมด", "BUY", "SELL"])
    if filter_type != "ทั้งหมด":
      df_display = df_display[df_display["Type"] == filter_type]

    # แสดงผลตารางสวยงาม
    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Entry": st.column_config.NumberColumn("Entry", format="$%.2f"),
            "SL": st.column_config.NumberColumn("SL", format="$%.2f"),
            "TP": st.column_config.NumberColumn("TP", format="$%.2f"),
        },
    )
  else:
    st.info("ยังไม่มีข้อมูลสัญญาณในระบบ")

with tab2:
  st.subheader("🛠️ การตั้งค่าระบบ (config.json)")
  config_file = "config.json"
  if os.path.exists(config_file):
    with open(config_file, "r", encoding="utf-8") as f:
      config_data = json.load(f)
    st.json(config_data)
  else:
    st.info("ไม่พบไฟล์ config.json (ใช้ค่า Default)")
    st.json({
        "rr_ratio": 2.0,
        "min_fvg_size": 0.5,
        "sl_buffer": 1.5,
        "status": "default",
    })
      
