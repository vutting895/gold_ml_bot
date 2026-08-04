# Gold ML Real-time Scanner Bot (Cloud Run Job)

โครงสร้างไฟล์โปรเจกต์สำหรับรันบอทสแกนราคาทองคำด้วย Machine Learning และส่งสัญญาณผ่าน Telegram

## 📁 ไฟล์ภายใน Zip:
- `gold_realtime_scanner.py` : สคริปต์ Python หลักสำหรับสแกนราคาทองคำ M5
- `gold_ml_filter.pkl` : ไฟล์โมเดล Machine Learning
- `Dockerfile` : คอนฟิกสำหรับสร้าง Container บน Cloud Run
- `requirements.txt` : รายชื่อไลบรารี Python ที่ใช้
- `deploy.sh` : สคริปต์อัตโนมัติสำหรับสั่ง Deploy ขึ้น GCP Cloud Run Job & Cloud Scheduler

## 🚀 วิธีใช้งานด่วน:
1. แตกไฟล์ Zip เข้าโฟลเดอร์โปรเจกต์
2. แก้ไข `TELEGRAM_TOKEN` และ `TELEGRAM_CHAT_ID` ในไฟล์ `gold_realtime_scanner.py`
3. รันสคริปต์ Deploy บน Google Cloud Shell:
   ```bash
   chmod +x deploy.sh
   ./deploy.sh
   ```
