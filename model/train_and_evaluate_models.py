import os
import glob
import time
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import joblib

# Class encoding
LINE_CODES = {'heart': 0, 'head': 1, 'life': 2, 'fate': 3}

# HSV Color Ranges
COLOR_RANGES = {
    'heart': ([0, 100, 100], [10, 255, 255]),
    'head': ([50, 100, 100], [70, 255, 255]),
    'life': ([130, 50, 50], [160, 255, 255]),
    'fate': ([100, 100, 100], [130, 255, 255])
}

# Extract arc length and area of largest contour
def extract_features(image_path, line_name):
    image = cv2.imread(image_path)
    if image is None:
        return None, None

    image = cv2.resize(image, (400, 400))  # resize for speed
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower, upper = COLOR_RANGES[line_name]
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        return cv2.arcLength(cnt, False), cv2.contourArea(cnt)
    return None, None

# Assign labels based on arc length and area
def assign_label(arc_length, area):
    if arc_length is None or area is None:
        return None
    if arc_length < 100:
        length_cat = 'short'
    elif arc_length < 200:
        length_cat = 'medium'
    else:
        length_cat = 'long'

    if area < 500:
        depth_cat = 'shallow'
    elif area < 1000:
        depth_cat = 'medium'
    else:
        depth_cat = 'deep'

    return f"{length_cat}_{depth_cat}"

# Load features and labels from dataset path
def load_dataset(dataset_dir):
    X, y = [], []

    extensions = ["*.jpg", "*.jpeg", "*.png"]
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(dataset_dir, "**", ext), recursive=True))

    if not image_paths:
        print(" No images found in dataset directory!")
        return [], []

    print(f"📁 Loaded {len(image_paths)} images from {dataset_dir}")
    for img_path in image_paths:
        for line in COLOR_RANGES.keys():
            arc_length, area = extract_features(img_path, line)
            label = assign_label(arc_length, area)
            if label:
                X.append([arc_length, area, LINE_CODES[line]])
                y.append(label)
    return X, y

# Main
if __name__ == "__main__":
    train_path = r"C:\Users\SHWETA\Desktop\Palmproject\data\archive"
    test_path = r"C:\Users\SHWETA\Desktop\Palmproject\data\MALE"

    print("📦 Loading training dataset...")
    X_train, y_train = load_dataset(train_path)

    print("🧪 Loading testing dataset...")
    X_test, y_test = load_dataset(test_path)

    if not X_train or not X_test:
        print("⚠️ Missing training or testing data.")
        exit()

    os.makedirs("model", exist_ok=True)

    # 🎯 Train Random Forest
    print("\n🌳 Training Random Forest...")
    start = time.time()
    rf_model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)
    print(f"✅ Random Forest Accuracy: {rf_acc * 100:.2f}%")
    joblib.dump(rf_model, "model/saved_rf_model.pkl")
    print(f"📁 Model saved to model/saved_rf_model.pkl ({time.time() - start:.2f}s)")

    # 🎯 Train Linear SVM
    print("\n🧠 Training Linear SVM...")
    start = time.time()
    svm_model = LinearSVC(max_iter=3000)
    svm_model.fit(X_train, y_train)
    svm_preds = svm_model.predict(X_test)
    svm_acc = accuracy_score(y_test, svm_preds)
    print(f"✅ Linear SVM Accuracy: {svm_acc * 100:.2f}%")
    joblib.dump(svm_model, "model/saved_svm_model.pkl")
    print(f"📁 Model saved to model/saved_svm_model.pkl ({time.time() - start:.2f}s)")
