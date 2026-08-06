from datetime import datetime, timedelta
import sys
import time
import pytz
import scanner


def get_seconds_until_next_m5_close(buffer_seconds=10):
  """คำนวณจำนวนวินาทีที่ต้องรอจนกว่าจะถึงเวลาปิดแท่ง M5 ถัดไป + Buffer ให้ API อัปเดตราคา"""
  tz_th = pytz.timezone("Asia/Bangkok")
  now = datetime.now(tz_th)

  # คำนวณนาทีของแท่ง M5 ถัดไป (0, 5, 10, 15, ..., 55)
  current_minute = now.minute
  next_m5_minute = ((current_minute // 5) + 1) * 5

  if next_m5_minute >= 60:
    next_run_time = now.replace(
        minute=0, second=buffer_seconds, microsecond=0
    ) + timedelta(hours=1)
  else:
    next_run_time = now.replace(
        minute=next_m5_minute, second=buffer_seconds, microsecond=0
    )

  wait_seconds = (next_run_time - now).total_seconds()

  # หากคำนวณแล้วค่าน้อยกว่าหรือเท่ากับ 0 ให้ขยับไปรอบ 5 นาทีถัดไป
  if wait_seconds <= 0:
    wait_seconds += 300

  return wait_seconds, next_run_time


def run_worker():
  tz_th = pytz.timezone("Asia/Bangkok")
  print(
      "🚀 [Render Worker] เริ่มต้นระบบ Gold SMC Scanner (M5 Sync Enabled)..."
  )
  sys.stdout.flush()

  while True:
    now_th = datetime.now(tz_th)
    day_of_week = now_th.weekday()  # 0=Monday, ..., 4=Friday, 5=Sat, 6=Sun

    # 1. ตรวจสอบวันทำการตลาดทองคำ (จันทร์ - ศุกร์)
    if day_of_week < 5:
      print(
          f"\n⏰ [{now_th.strftime('%Y-%m-%d %H:%M:%S')} UTC+7] กำลังเริ่มสแกนราคา..."
      )
      sys.stdout.flush()

      try:
        scanner.main()
      except Exception as e:
        print(f"❌ [Worker Error] เกิดข้อผิดพลาดขณะรันการสแกน: {e}")
        sys.stdout.flush()
    else:
      print(
          f"\n😴 [{now_th.strftime('%Y-%m-%d %H:%M:%S')} UTC+7] ตลาดทองคำปิดทำการ (วันเสาร์-อาทิตย์)"
      )
      sys.stdout.flush()

    # 2. คำนวณเวลารอจนกว่าจะปิดแท่ง M5 ถัดไป
    wait_sec, next_time = get_seconds_until_next_m5_close(buffer_seconds=10)
    print(
        f"⏳ รอบสแกนถัดไปเวลา: {next_time.strftime('%H:%M:%S')} UTC+7 (รออีก"
        f" {int(wait_sec)} วินาที)"
    )
    sys.stdout.flush()

    # 3. หน่วงเวลารอ
    time.sleep(wait_sec)


if __name__ == "__main__":
  run_worker()
  
