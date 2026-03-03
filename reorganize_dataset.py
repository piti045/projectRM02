"""
🧹 สคริปต์เพื่อทำความสะอาด dataset เก่า
และให้มีการเก็บข้อมูลใหม่ที่ถูกต้อง
"""

import os
import shutil

DATA_PATH = "dataset"

print("=" * 60)
print("⚠️  การลบข้อมูล dataset เก่า")
print("=" * 60)

if os.path.exists(DATA_PATH):
    # Backup โฟลเดอร์เก่า
    backup_path = DATA_PATH + "_backup"
    if os.path.exists(backup_path):
        shutil.rmtree(backup_path)
    
    shutil.move(DATA_PATH, backup_path)
    print(f"✅ Backed up to: {backup_path}")
    
    # สร้างโฟลเดอร์ใหม่
    os.makedirs(DATA_PATH)
    print(f"✅ New empty {DATA_PATH} folder created")

print("\n" + "=" * 60)
print("📋 ขั้นตอนต่อไป:")
print("=" * 60)
print("""
1. รัน collect_data.py เพื่อเก็บข้อมูลใหม่:
   python collect_data.py

2. เมื่อเสร็จแล้วตรวจสอบด้วย:
   python debug_dataset.py

3. เทรนด้วย:
   python train_model.py

4. ทดสอบใน:
   python app.py
""")
