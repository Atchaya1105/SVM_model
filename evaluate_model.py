import os
import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    ConfusionMatrixDisplay
)

# =========================
# CONFIG
# =========================
DATASET_PATH = "../dataset/OriginalDataset"
MODEL_PATH = "svm_model_fast.pkl"
IMG_SIZE = 128

CATEGORIES = {
    'MildDemented': 0,
    'ModerateDemented': 1,
    'NonDemented': 2,
    'VeryMildDemented': 3
}

# =========================
# LOAD DATA
# =========================
X, y = [], []

print("📂 Loading images for evaluation...")

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

# =========================
# TRAIN TEST SPLIT (same as training)
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# LOAD MODEL
# =========================
print("📦 Loading trained model...")
model = joblib.load(MODEL_PATH)

# =========================
# PREDICTION
# =========================
y_pred = model.predict(X_test)

# =========================
# ACCURACY
# =========================
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"\n🎯 Accuracy: {accuracy:.2f}%")

# =========================
# CLASSIFICATION REPORT
# =========================
print("\n📊 Classification Report:\n")
print(classification_report(
    y_test,
    y_pred,
    target_names=CATEGORIES.keys()
))

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CATEGORIES.keys()
)

disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Alzheimer Classification")
plt.savefig("confusion_matrix.png")
plt.show()

print("✅ Confusion matrix saved as confusion_matrix.png")
