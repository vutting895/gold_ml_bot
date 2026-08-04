import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="Gold SMC Forward Test Dashboard", layout="wide")

st.title("🏆 XAU/USD Real-time Forward Test Performance")
st.markdown("ระบบติดตามผลการเทรดจริง Real-time SMC + ML Filter")

# อ่านข้อมูลจาก Google Sheets
@st.cache_data(ttl=60) # Cache ข้อมูล 60 วินาที
def load_data():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds_dict = st.secrets["GOOGLE_CREDENTIALS"]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
    client = gspread.authorize(creds)
    sheet = client.open("Gold_Trading_Logs").worksheet("Signals") #[span_1](start_span)[span_1](end_span)
    data = sheet.get_all_records()
    return pd.DataFrame(data)

try:
    df = load_data()

    if not df.empty and 'Status' in df.columns and 'PnL' in df.columns:
        # คำนวณ Metrics
        total_trades = len(df)
        closed_trades = df[df['Status'] != 'OPEN']
        wins = len(df[df['Status'].str.contains('WIN', na=False)])
        losses = len(df[df['Status'].str.contains('LOSS', na=False)])
        be_trades = len(df[df['Status'].str.contains('BE', na=False)])
        
        win_rate = (wins / len(closed_trades) * 100) if len(closed_trades) > 0 else 0
        total_pnl = df['PnL'].sum()

        # แสดง Metrics Cards
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Signals", total_trades)
        col2.metric("Win Rate", f"{win_rate:.1f}%")
        col3.metric("Wins / Losses / BE", f"{wins} / {losses} / {be_trades}")
        col4.metric("Net PnL ($)", f"${total_pnl:,.2f}", delta_color="normal")
        col5.metric("Active Trades (OPEN)", len(df[df['Status'] == 'OPEN']))

        st.divider()

        # แสดงตารางข้อมูล
        st.subheader("📋 Signal History & Status")
        st.dataframe(df.sort_index(ascending=False), use_container_width=True)

    else:
        st.info("ยังไม่มีข้อมูลสัญญาณในระบบ หรือโครงสร้างคอลัมน์ยังไม่ครบถ้วน (ต้องการคอลัมน์ Status และ PnL)")

except Exception as e:
    st.error(f"เกิดข้อผิดพลาดในการโหลดข้อมูล: {e}")
    
