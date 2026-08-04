import json
import os
import requests
import pandas as pd

TWELVE_DATA_API_KEY = os.environ.get("TWELVE_DATA_API_KEY")
CONFIG_FILE = "config.json"


def run_optimization():
  print("กำลังรัน Auto Backtest Optimizer...")

  # สมมติการค้นหาค่า R:R Ratio และ FVG Min Size ที่ให้ Win Rate ดีที่สุด
  best_config = {
      "rr_ratio": 2.0,
      "min_fvg_size": 0.5,
      "sl_buffer": 1.5,
      "last_updated": str(pd.Timestamp.now(tz="Asia/Bangkok")),
  }

  with open(CONFIG_FILE, "w") as f:
    json.dump(best_config, f, indent=4)

  print(f"✨ ปรับแต่งและบันทึกค่า Config สำเร็จลงไฟล์ '{CONFIG_FILE}'")


if __name__ == "__main__":
  run_optimization()
    
