# Week 2 - Classification, Metrics & Model Selection

**ทฤษฎี (2 ชม.):** การแยกประเภทข้อมูล, การวัดผล (Precision, Recall, ROC Curve), ปัญหา Overfitting/Underfitting

---

## Workshop 1: การสร้างโมเดล Binary Classification

เรียนรู้การสร้างโมเดลจำแนกข้อมูลที่มีเพียง 2 คลาส (เช่น เป็นโรค หรือ ไม่เป็นโรค) ด้วยอัลกอริทึม `LogisticRegression`

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. โหลดชุดข้อมูลผู้ป่วยมะเร็งเต้านม
data = load_breast_cancer()
X, y = data.data, data.target

# 2. แบ่งข้อมูล Train/Test (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. สร้างและฝึกโมเดล Logistic Regression
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# 4. ทำนายผลและวัดความแม่นยำ (Accuracy)
y_pred = model.predict(X_test)
print(f"Accuracy ของโมเดล: {accuracy_score(y_test, y_pred):.4f} หรือ {accuracy_score(y_test, y_pred)*100:.2f}%")
```

---

## Workshop 2: การจัดการข้อมูลแบบ Multiclass และสร้าง Confusion Matrix

การจำแนกข้อมูลที่มีมากกว่า 2 คลาส และการดูว่าโมเดลทำนาย "สับสน" ที่คลาสไหนด้วย Confusion Matrix

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. โหลดข้อมูลดอก Iris (มี 3 สายพันธุ์)
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)

# 2. สร้างโมเดลแบบ Decision Tree
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# 3. ดูรายงานผลเชิงลึก (Precision, Recall, F1-Score)
print("--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 4. พล็อต Confusion Matrix เพื่อดูจุดที่โมเดลทายพลาด
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted (สิ่งที่โมเดลทาย)')
plt.ylabel('Actual (ความจริง)')
plt.title('Confusion Matrix')
plt.show()
```

---

## Workshop 3: การทำ Cross-validation และ Hyperparameter Tuning

การค้นหาค่าพารามิเตอร์ที่ดีที่สุดให้โมเดลแบบอัตโนมัติด้วย `GridSearchCV` เพื่อป้องกันโมเดลจำข้อมูลมากเกินไป (Overfitting)

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# 1. กำหนดช่วงของ Hyperparameter ที่ต้องการให้ AI สุ่มหา
param_grid = {
    'n_estimators': [50, 100, 200],  # จำนวนต้นไม้ในป่า
    'max_depth': [None, 10, 20]      # ความลึกสูงสุดของต้นไม้
}

# 2. สร้างระบบ GridSearch ที่ใช้ 5-Fold Cross Validation
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42), 
    param_grid=param_grid, 
    cv=5,      # ตัดแบ่งข้อมูล 5 ส่วนเพื่อเทสต์สลับกัน
    n_jobs=-1  # ใช้ CPU ทุกคอร์ที่มีให้ประมวลผลเร็วขึ้น
)

# 3. เริ่มรันหาค่าพารามิเตอร์ที่เจ๋งที่สุด
grid_search.fit(X_train, y_train)

print("ค่าพารามิเตอร์ที่ดีที่สุด (Best Parameters):", grid_search.best_params_)
print("ความแม่นยำเฉลี่ยที่ดีที่สุด (Best CV Score):", round(grid_search.best_score_, 4))
```

---

## Workshop 4: การใช้ Support Vector Machines (SVM) และเปรียบเทียบประสิทธิภาพ

เปรียบเทียบผลลัพธ์ของโมเดล SVM แบบเส้นตรง (Linear Kernel) กับแบบโค้งงอ (RBF Kernel) สำหรับข้อมูลที่มีความซับซ้อน

```python
from sklearn.svm import SVC

# ลองสร้าง SVM Kernel=Linear (ใช้เส้นตรงแบ่งข้อมูล)
svm_linear = SVC(kernel='linear')
svm_linear.fit(X_train, y_train)
acc_linear = svm_linear.score(X_test, y_test)

# ลองสร้าง SVM Kernel=RBF (จัดการแบ่งข้อมูลที่ซับซ้อน/โค้งงอได้)
svm_rbf = SVC(kernel='rbf')
svm_rbf.fit(X_train, y_train)
acc_rbf = svm_rbf.score(X_test, y_test)

print(f"ความแม่นยำ SVM (Linear Kernel): {acc_linear:.4f}")
print(f"ความแม่นยำ SVM (RBF Kernel): {acc_rbf:.4f}")
```
