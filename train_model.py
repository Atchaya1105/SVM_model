import os
import cv2
import numpy as np
import joblib
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# ========================
# CONFIG
# =========================
DATASET_PATH = "../dataset/OriginalDataset"
IMG_SIZE = 128   # 🔥 Reduced size (VERY IMPORTANT)

CATEGORIES = {
    'MildDemented': 0,
    'ModerateDemented': 1,
    'NonDemented': 2,
    'VeryMildDemented': 3
}

# =========================
# LOAD DATA
# =========================
X = []
y = []

print("📂 Loading images...")

for category, label in CATEGORIES.items():
    folder_path = os.path.join(DATASET_PATH, category)
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path, 0)

        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img.flatten()
            X.append(img)
            y.append(label)

X = np.array(X, dtype=np.float32)
y = np.array(y)

print("✅ Images loaded:", X.shape)

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# MODEL PIPELINE
# =========================
model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", LinearSVC(max_iter=5000))
])

# =========================
# TRAIN
# =========================
print("🚀 Training started...")
model.fit(X_train, y_train)
print("✅ Training completed")

# =========================
# EVALUATE
# =========================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"🎯 Accuracy: {accuracy:.2f}%")

# =========================
# SAVE MODEL
# =========================
joblib.dump(model, "svm_model_fast.pkl")
print("💾 Model saved as svm_model_fast.pkl")
