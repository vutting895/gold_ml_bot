import json

def optimize():
    print("⚙️ กำลังทำการ Optimize ค่าพารามิเตอร์...")
    # โค้ด Grid Search / Backtest
    best_params = {
        "PROBA_THRESHOLD": 0.60,
        "RR_RATIO": 2.0,
        "GOLD_SL_BUFFER": 3.0
    }
    
    with open("best_config.json", "w") as f:
        json.dump(best_params, f, indent=4)
        
    print("✅ อัปเดตไฟล์ best_config.json เรียบร้อยแล้ว")

if __name__ == "__main__":
    optimize()
