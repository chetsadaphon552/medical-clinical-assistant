# ใช้ Python 3.10 เป็นฐาน (เบาและเสถียรสำหรับ AI)
FROM python:3.10-slim

# ตั้งค่า Working Directory ใน Container
WORKDIR /app

# คัดลอกไฟล์ requirements.txt มาก่อนเพื่อติดตั้ง Dependencies (ทำ Caching)
COPY requirements.txt .

# ติดตั้ง Dependencies แบบไม่เก็บ Cache เพื่อลดขนาด Image
RUN pip install --no-cache-dir -r requirements.txt

# คัดลอกไฟล์ทั้งหมดในโปรเจกต์เข้ามาใน Container
COPY . .

# สร้างโฟลเดอร์สำหรับเก็บ log เผื่อไว้
RUN mkdir -p logs

# คำสั่งสำหรับรันตอน Start Container
CMD ["python", "demo_interactive.py"]
