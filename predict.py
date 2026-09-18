import sys
import os
import cv2
import numpy as np
import joblib

# TensorFlow / MobileNetV2
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")


# ============================================================
# LOAD CNN FEATURE EXTRACTOR
# ============================================================

print("Loading MobileNetV2...")

cnn_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(224, 224, 3)
)

cnn_model.trainable = False

print("MobileNetV2 loaded successfully.")
print("CNN feature size:", cnn_model.output_shape[-1])


# ============================================================
# LOAD CLASSIFIERS
# ============================================================

print("\nLoading classifiers...")

svm_model = joblib.load(
    os.path.join(MODEL_DIR, "svm_cnn_model.pkl")
)

knn_model = joblib.load(
    os.path.join(MODEL_DIR, "knn_cnn_model.pkl")
)

decision_tree_model = joblib.load(
    os.path.join(MODEL_DIR, "decision_tree_cnn_model.pkl")
)

random_forest_model = joblib.load(
    os.path.join(MODEL_DIR, "random_forest_cnn_model.pkl")
)

scaler = joblib.load(
    os.path.join(MODEL_DIR, "cnn_scaler.pkl")
)

class_labels = joblib.load(
    os.path.join(MODEL_DIR, "class_labels_cnn.pkl")
)

print("All CNN classifiers loaded successfully.")

print("\nClasses:")
for i, label in enumerate(class_labels):
    print(f"{i}: {label}")


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):
    """
    Prepare image for MobileNetV2.
    """

    # Resize to MobileNetV2 input size
    image = cv2.resize(
        image,
        (224, 224),
        interpolation=cv2.INTER_AREA
    )

    # OpenCV BGR -> RGB
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # Convert to float32
    image = image.astype(np.float32)

    # MobileNetV2 preprocessing
    image = preprocess_input(image)

    return image


# ============================================================
# CNN FEATURE EXTRACTION
# ============================================================

def extract_cnn_features(image):
    """
    Extract 1280-dimensional MobileNetV2 feature vector.
    """

    processed_image = preprocess_image(image)

    # Add batch dimension
    image_batch = np.expand_dims(
        processed_image,
        axis=0
    )

    # Extract CNN features
    features = cnn_model.predict(
        image_batch,
        verbose=0
    )

    features = features.reshape(1, -1)

    return features


# ============================================================
# MAJORITY VOTING
# ============================================================

def majority_vote(predictions):
    """
    Perform hard majority voting.
    """

    predictions = np.array(predictions)

    unique_classes, counts = np.unique(
        predictions,
        return_counts=True
    )

    winner = unique_classes[np.argmax(counts)]

    votes = np.max(counts)

    return winner, votes


# ============================================================
# PREDICTION
# ============================================================

def predict_fruit(image_path):

    print("\n" + "=" * 60)
    print("FRUIT CLASSIFICATION")
    print("=" * 60)

    print("\nImage:", image_path)

    # --------------------------------------------------------
    # Check image
    # --------------------------------------------------------

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            "Unable to read the image."
        )

    print(
        "Original image size:",
        image.shape[1],
        "x",
        image.shape[0]
    )

    # --------------------------------------------------------
    # CNN feature extraction
    # --------------------------------------------------------

    print("\nExtracting MobileNetV2 features...")

    features = extract_cnn_features(image)

    print(
        "CNN feature shape:",
        features.shape
    )

    # Safety check
    if features.shape[1] != 1280:
        raise ValueError(
            f"Expected 1280 CNN features, "
            f"but received {features.shape[1]}"
        )

    # --------------------------------------------------------
    # Scaling
    # --------------------------------------------------------

    features_scaled = scaler.transform(features)

    print(
        "Scaled feature shape:",
        features_scaled.shape
    )

    # --------------------------------------------------------
    # Individual classifier predictions
    # --------------------------------------------------------

    svm_prediction = svm_model.predict(
        features_scaled
    )[0]

    knn_prediction = knn_model.predict(
        features_scaled
    )[0]

    decision_tree_prediction = decision_tree_model.predict(
        features_scaled
    )[0]

    random_forest_prediction = random_forest_model.predict(
        features_scaled
    )[0]

    # --------------------------------------------------------
    # Convert numeric predictions to class names
    # --------------------------------------------------------

    svm_label = class_labels[int(svm_prediction)]
    knn_label = class_labels[int(knn_prediction)]
    dt_label = class_labels[int(decision_tree_prediction)]
    rf_label = class_labels[int(random_forest_prediction)]

    # --------------------------------------------------------
    # Display predictions
    # --------------------------------------------------------

    print("\n" + "-" * 60)
    print("INDIVIDUAL CLASSIFIER PREDICTIONS")
    print("-" * 60)

    print(f"SVM             : {svm_label}")
    print(f"KNN             : {knn_label}")
    print(f"Decision Tree   : {dt_label}")
    print(f"Random Forest   : {rf_label}")

    # --------------------------------------------------------
    # Majority voting
    # --------------------------------------------------------

    predictions = [
        svm_label,
        knn_label,
        dt_label,
        rf_label
    ]

    final_prediction, votes = majority_vote(
        predictions
    )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL PREDICTION")
    print("=" * 60)

    print(f"\nFruit       : {final_prediction}")
    print(f"Votes       : {votes}/4")

    if votes >= 3:
        confidence_text = "Strong agreement"
    elif votes == 2:
        confidence_text = "Majority agreement"
    else:
        confidence_text = "No clear majority"

    print(f"Agreement   : {confidence_text}")

    print("\n" + "=" * 60)

    return {
        "svm": svm_label,
        "knn": knn_label,
        "decision_tree": dt_label,
        "random_forest": rf_label,
        "final_prediction": final_prediction,
        "votes": int(votes)
    }


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("\nUsage:")
        print("python predict.py <image_path>")
        print("\nExample:")
        print("python predict.py test_images/apple.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    try:
        predict_fruit(image_path)

    except Exception as e:
        print("\nERROR:")
        print(e)
        sys.exit(1)