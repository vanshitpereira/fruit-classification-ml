from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys

# Allow importing predict.py from project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from predict import predict_fruit


app = Flask(__name__)
CORS(app)


# ============================================================
# HOME
# ============================================================

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "success",
        "message": "Fruit Classification API is running",
        "model": "MobileNetV2 + Multiple Classifiers",
        "classes": 7
    })


# ============================================================
# PREDICTION
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    # Check image
    if "image" not in request.files:
        return jsonify({
            "status": "error",
            "message": "No image uploaded"
        }), 400

    image = request.files["image"]

    if image.filename == "":
        return jsonify({
            "status": "error",
            "message": "No image selected"
        }), 400

    # Create temporary upload folder
    upload_folder = os.path.join(
        PROJECT_ROOT,
        "backend",
        "uploads"
    )

    os.makedirs(upload_folder, exist_ok=True)

    # Save image
    image_path = os.path.join(
        upload_folder,
        image.filename
    )

    image.save(image_path)

    try:

        # Run actual ML prediction
        prediction = predict_fruit(image_path)

        return jsonify({
            "status": "success",
            "filename": image.filename,
            "svm": prediction["svm"],
            "knn": prediction["knn"],
            "decision_tree": prediction["decision_tree"],
            "random_forest": prediction["random_forest"],
            "final_prediction": prediction["final_prediction"],
            "votes": prediction["votes"]
        })

    except Exception as e:

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

    finally:

        # Delete uploaded image after prediction
        if os.path.exists(image_path):
            os.remove(image_path)


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )