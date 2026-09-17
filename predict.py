# ============================================================
# FRUIT CLASSIFICATION USING MULTIPLE CLASSIFIERS
# New Image Prediction
# ============================================================

import sys
import os
import cv2
import numpy as np
import joblib

from skimage.feature import hog, local_binary_pattern


# ============================================================
# 1. FEATURE EXTRACTION
# Same method used during training
# Color + HOG + LBP
# ============================================================

def extract_features(image):

    features = []

    # --------------------------------------------------------
    # 1. COLOR FEATURES
    # --------------------------------------------------------

    # Convert RGB image to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # RGB mean and standard deviation
    rgb_mean = np.mean(image, axis=(0, 1))
    rgb_std = np.std(image, axis=(0, 1))

    # HSV mean and standard deviation
    hsv_mean = np.mean(hsv, axis=(0, 1))
    hsv_std = np.std(hsv, axis=(0, 1))

    features.extend(rgb_mean)
    features.extend(rgb_std)
    features.extend(hsv_mean)
    features.extend(hsv_std)

    # --------------------------------------------------------
    # 2. COLOR HISTOGRAM
    # --------------------------------------------------------

    for channel in range(3):

        hist = cv2.calcHist(
            [image],
            [channel],
            None,
            [16],
            [0, 256]
        )

        hist = cv2.normalize(hist, hist).flatten()

        features.extend(hist)

    # --------------------------------------------------------
    # 3. HOG FEATURES
    # --------------------------------------------------------

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )

    features.extend(hog_features)

    # --------------------------------------------------------
    # 4. LBP TEXTURE FEATURES
    # --------------------------------------------------------

    radius = 1
    n_points = 8 * radius

    lbp = local_binary_pattern(
        gray,
        n_points,
        radius,
        method='uniform'
    )

    n_bins = n_points + 2

    lbp_hist, _ = np.histogram(
        lbp.ravel(),
        bins=n_bins,
        range=(0, n_bins)
    )

    # Normalize LBP histogram
    lbp_hist = lbp_hist.astype("float32")
    lbp_hist /= (lbp_hist.sum() + 1e-7)

    features.extend(lbp_hist)

    return np.array(features, dtype=np.float32)


# ============================================================
# 2. LOAD MODELS
# ============================================================

MODEL_DIR = "models"

print("\nLoading trained models...")

svm_model = joblib.load(
    os.path.join(MODEL_DIR, "svm_model.pkl")
)

knn_model = joblib.load(
    os.path.join(MODEL_DIR, "knn_model.pkl")
)

dt_model = joblib.load(
    os.path.join(MODEL_DIR, "decision_tree_model.pkl")
)

rf_model = joblib.load(
    os.path.join(MODEL_DIR, "random_forest_model.pkl")
)

scaler = joblib.load(
    os.path.join(MODEL_DIR, "scaler.pkl")
)

class_labels = joblib.load(
    os.path.join(MODEL_DIR, "class_labels.pkl")
)

print("Models loaded successfully!")


# ============================================================
# 3. CHECK IMAGE PATH
# ============================================================

if len(sys.argv) < 2:

    print("\nUsage:")
    print("python predict.py path/to/image.jpg")

    sys.exit(1)


image_path = sys.argv[1]


if not os.path.exists(image_path):

    print("\nERROR: Image file not found!")
    print("Path:", image_path)

    sys.exit(1)


# ============================================================
# 4. LOAD IMAGE
# ============================================================

print("\nLoading image...")

image_bgr = cv2.imread(image_path)

if image_bgr is None:

    print("ERROR: Could not read the image.")

    sys.exit(1)


# OpenCV loads images as BGR.
# Convert to RGB because the training pipeline
# expects RGB images.

image_rgb = cv2.cvtColor(
    image_bgr,
    cv2.COLOR_BGR2RGB
)


# ============================================================
# 5. RESIZE IMAGE
# ============================================================

image_rgb = cv2.resize(
    image_rgb,
    (100, 100)
)


print("Image shape:", image_rgb.shape)


# ============================================================
# 6. EXTRACT FEATURES
# ============================================================

print("\nExtracting features...")

features = extract_features(image_rgb)

print("Feature vector shape:", features.shape)

if features.shape[0] != 4426:

    print(
        "\nWARNING: Expected 4426 features, "
        "but got", features.shape[0]
    )


# Convert to 2D array

features = features.reshape(1, -1)


# ============================================================
# 7. APPLY SAME SCALER USED DURING TRAINING
# ============================================================

print("Applying feature scaling...")

features_scaled = scaler.transform(features)

print("Scaled feature shape:", features_scaled.shape)


# ============================================================
# 8. PREDICTION FROM EACH MODEL
# ============================================================

print("\nGenerating predictions...")

svm_prediction = svm_model.predict(features_scaled)[0]

knn_prediction = knn_model.predict(features_scaled)[0]

dt_prediction = dt_model.predict(features_scaled)[0]

rf_prediction = rf_model.predict(features_scaled)[0]


# ============================================================
# 9. CONVERT CLASS INDEX TO CLASS NAME
# ============================================================

def get_class_name(prediction):

    try:

        return class_labels[prediction]

    except:

        return str(prediction)


svm_result = get_class_name(svm_prediction)

knn_result = get_class_name(knn_prediction)

dt_result = get_class_name(dt_prediction)

rf_result = get_class_name(rf_prediction)


# ============================================================
# 10. MAJORITY VOTING
# ============================================================

predictions = [
    svm_prediction,
    knn_prediction,
    dt_prediction,
    rf_prediction
]


unique_classes, counts = np.unique(
    predictions,
    return_counts=True
)

final_prediction = unique_classes[
    np.argmax(counts)
]


final_result = get_class_name(
    final_prediction
)


# ============================================================
# 11. DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 55)
print("          FRUIT CLASSIFICATION RESULT")
print("=" * 55)

print("\nSVM              :", svm_result)

print("KNN              :", knn_result)

print("Decision Tree    :", dt_result)

print("Random Forest    :", rf_result)

print("\n" + "-" * 55)

print("FINAL PREDICTION :", final_result)

print("-" * 55)

print("\nVoting:")
print("SVM              :", svm_result)
print("KNN              :", knn_result)
print("Decision Tree    :", dt_result)
print("Random Forest    :", rf_result)

print("\n" + "=" * 55)