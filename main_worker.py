# main_worker.py
import logging
import time
import scanner

# ตั้งค่า Logging ระบบ
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def run_worker():
    logging.info("🤖 เริ่มต้นทำงาน main_worker.py สำหรับ Gold SMC Scanner...")

    while True:
        try:
            logging.info("🔍 เริ่มการสแกนตลาดทองคำ (XAU/USD)...")
            scanner.main()
            logging.info("✅ รอบการสแกนเสร็จสิ้น รอรอบถัดไปในอีก 5 นาที...")
        except Exception as e:
            logging.error(f"❌ เกิดข้อผิดพลาดใน Worker Loop: {e}", exc_info=True)

        # หน่วงเวลา 300 วินาที (5 นาที) ให้สอดคล้องกับแท่งเทียน Timeframe M5
        time.sleep(300)


if __name__ == "__main__":
    run_worker()
  
