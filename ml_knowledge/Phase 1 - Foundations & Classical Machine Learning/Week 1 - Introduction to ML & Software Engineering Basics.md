# Week 1 - Introduction to ML & Software Engineering Basics

**ทฤษฎี (2 ชม.):** ภาพรวมของ ML, กระบวนการทำโปรเจกต์ ML แบบ End-to-End, แนวปฏิบัติที่ดีในการเขียนโค้ด (Git, Environment)

---

## Workshop 1: การตั้งค่า Environment และ Git Version Control

การตั้งค่าสภาพแวดล้อม (Environment) เป็นสิ่งสำคัญในการป้องกันปัญหา Dependency หรือเวอร์ชันของไลบรารีขัดแย้งกัน

**1. สร้าง Environment ด้วย Conda หรือ Venv**
```bash
# สำหรับ Conda
conda create -n ml_env python=3.10
conda activate ml_env

# สำหรับ Venv (Python พื้นฐาน)
python -m venv ml_env
# เปิดใช้งานใน Windows (Command Prompt)
ml_env\Scripts\activate
```

**2. การใช้ Git พื้นฐาน**
```bash
git init                  # เริ่มต้นระบบ Git ในโฟลเดอร์
git add .                 # เพิ่มไฟล์ทั้งหมดเตรียมบันทึก (Staging)
git commit -m "init"      # บันทึกเวอร์ชันโค้ด
git push origin main      # อัปโหลดโค้ดขึ้น GitHub
```

---

## Workshop 2: Data Manipulation พื้นฐานด้วย Pandas & NumPy

ไลบรารีพื้นฐานที่ Data Scientist และ ML Engineer ต้องใช้สำหรับการจัดการตัวเลขและตารางข้อมูล

**การจัดการข้อมูลด้วย NumPy**
```python
import numpy as np

# สร้าง Array (เวกเตอร์)
arr = np.array([1, 2, 3, 4, 5])
print("Mean:", np.mean(arr)) # หาค่าเฉลี่ย
print("Sum:", np.sum(arr))   # หาผลรวม
```

**การจัดการข้อมูลตารางด้วย Pandas**
```python
import pandas as pd

# สร้าง DataFrame จาก Dictionary
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'Score': [85, 90, 88]
}
df = pd.DataFrame(data)

# ดูข้อมูลเบื้องต้น
print(df.head())

# ดูสถิติพื้นฐาน (เฉลี่ย, สูงสุด, ต่ำสุด)
print(df.describe())
```

---

## Workshop 3: การทำ Data Cleaning และ EDA เบื้องต้น

การสำรวจข้อมูล (Exploratory Data Analysis) และทำความสะอาดข้อมูล (Clean) ก่อนนำไปใช้ฝึกโมเดล

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# โหลดชุดข้อมูลตัวอย่าง (Titanic dataset ข้อมูลผู้โดยสารเรือไททานิค)
df = sns.load_dataset('titanic')

# 1. การตรวจสอบค่าว่าง (Missing Values)
print("จำนวนค่าว่างในแต่ละคอลัมน์:\n", df.isnull().sum())

# 2. การจัดการค่าว่าง (เติมค่าเฉลี่ยให้คอลัมน์อายุ 'age')
df['age'].fillna(df['age'].mean(), inplace=True)

# 3. EDA - พล็อตดูกราฟอัตราการรอดชีวิตแยกตามเพศ
sns.countplot(data=df, x='survived', hue='sex')
plt.title("Survival by Gender on Titanic")
plt.show()
```

---

## Workshop 4: สร้างโมเดลแรกด้วย Scikit-Learn (Linear Regression)

เราจะใช้ Scikit-Learn สร้างโมเดลทำนายตัวเลขแบบง่ายๆ (Regression)

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# 1. สร้างข้อมูลจำลอง (พื้นที่บ้าน -> ราคาบ้าน)
X = np.array([[50], [60], [70], [80], [100]]) # ฟีเจอร์: พื้นที่ (ตร.ม.)
y = np.array([1.5, 1.8, 2.1, 2.4, 3.0])       # เป้าหมาย: ราคา (ล้านบาท)

# 2. แบ่งข้อมูลสำหรับ Train 80% และ Test 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. สร้างและฝึกโมเดล (Train)
model = LinearRegression()
model.fit(X_train, y_train)

# 4. ลองทำนายผล (Predict)
predictions = model.predict(X_test)
print(f"ราคาที่ทำนายได้: {predictions[0]:.2f} ล้านบาท")
print(f"ราคาจริง: {y_test[0]:.2f} ล้านบาท")

# 5. วัดผลความแม่นยำด้วย MSE (ยิ่งน้อยยิ่งดี)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse:.2f}")
```
