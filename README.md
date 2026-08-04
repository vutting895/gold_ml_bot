# 🤖 Automated Gold (XAU/USD) SMC + ML Scanner System

ระบบสแกนราคาทองคำอัตโนมัติบนไทม์เฟรม M5 ตามกลยุทธ์ **Smart Money Concept (SMC: Wave 3 + Fair Value Gap)** ร่วมกับการกรองสัญญาณความน่าจะเป็นด้วย **Machine Learning (Random Forest Classifier)** พร้อมระบบส่งสัญญาณเข้า Telegram และ Auto-Retrain อัตโนมัติผ่าน GitHub Actions

---

## 🌟 ฟีเจอร์หลัก (Key Features)

* **Multi-Timeframe Analysis:** วิเคราะห์แนวโน้มใหญ่ร่วมกันระหว่าง H1 (EMA 50), M15 (EMA 20) และแท่งเทียน M5
* **SMC Signal Detection:** ตรวจจับพื้นที่เกิด Fair Value Gap (FVG) เพื่อหาจุดเข้าสถิติต่ำ Risk-to-Reward สูง
* **ML Quality Filter:** ใช้โมเดล Machine Learning คัดกรองสัญญาณที่มีความน่าจะเป็นชนะ (Win Probability) สูงกว่าเกณฑ์ที่กำหนด
* **Dynamic Position Sizing:** คำนวณขนาด Lot Size อัตโนมัติอิงตามความเสี่ยงพอร์ต (% Risk) และปรับเพิ่มความเสี่ยงเมื่อโมเดลมีความมั่นใจสูง
* **Auto Parameter Optimization:** ระบบ Grid Search Backtest ค้นหาพารามิเตอร์ R:R Ratio, Stop Loss Buffer และ Probability Threshold ที่ดีที่สุดย้อนหลังอัตโนมัติ
* **Fully Automated via GitHub Actions:** 
  * สแกนตลาดฟรีทุกๆ 5 นาที (ไม่ต้องเปิดคอมทิ้งไว้)
  * เทรนโมเดล ML ใหม่และอัปเดตไฟล์คอนฟิกอัตโนมัติทุกสัปดาห์

---

## 📂 โครงสร้างโฟลเดอร์โปรเจกต์ (Project Structure)

```text
.
├── .github/
│   └── workflows/
│       ├── gold_scanner.yml          # [Workflow] สแกนราคา Real-time ทุก 5 นาที
│       └── retrain_and_optimize.yml  # [Workflow] เทรนโมเดล & หา Config ใหม่ทุกวันอาทิตย์
├── .gitignore                        # ไฟล์ยกเว้นขยะ/แคช
├── README.md                         # คู่มือการใช้งานระบบ
├── requirements.txt                  # รายชื่อ Python Dependencies
├── best_config.json                  # ไฟล์เก็บค่า Config ที่ได้จากการ Auto-Optimize
├── gold_ml_filter.pkl                # ไฟล์โมเดล Machine Learning
├── auto_backtest_optimizer.py        # สคริปต์ Backtest & Grid Search
├── train_gold_model.py               # สคริปต์เทรนโมเดล ML
├── scanner.py                        # สคริปต์ Real-time Scanner หลัก
├── gold_realtime_scanner.py          # สคริปต์ Real-time Scanner (เวอร์ชันสมบูรณ์)
└── deploy.sh                         # สคริปต์ Shell สำหรับ Auto Git Commit & Push
