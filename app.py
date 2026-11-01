
import json
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from ai_edge_litert.interpreter import Interpreter

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "arecanut_model.tflite"
CLASSES_PATH = BASE_DIR / "models" / "class_names.json"

MAX_IMAGE_SIZE = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

app.config["MAX_CONTENT_LENGTH"] = MAX_IMAGE_SIZE

# Load model and labels once when the service starts.
with open(CLASSES_PATH, "r", encoding="utf-8") as file:
    CLASS_NAMES = json.load(file)

interpreter = Interpreter(model_path=str(MODEL_PATH))
interpreter.allocate_tensors()

INPUT_DETAILS = interpreter.get_input_details()
OUTPUT_DETAILS = interpreter.get_output_details()


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service": "AgriSpectra Arecanut Disease API",
        "status": "running"
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "success": True,
        "model": "MobileNetV2 TFLite",
        "categories": CLASS_NAMES
    })


@app.route("/predict-arecanut", methods=["POST"])
def predict_arecanut():
    if "image" not in request.files:
        return jsonify({
            "success": False,
            "error": "Please upload an image using the image field."
        }), 400

    image_file = request.files["image"]

    if not image_file.filename:
        return jsonify({
            "success": False,
            "error": "No image selected."
        }), 400

    filename = secure_filename(image_file.filename)

    if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        return jsonify({
            "success": False,
            "error": "Unsupported image format."
        }), 400

    try:
        image = Image.open(image_file.stream).convert("RGB")
        image = image.resize((224, 224))

        input_data = np.asarray(image, dtype=np.float32)
        input_data = np.expand_dims(input_data, axis=0)

        interpreter.set_tensor(
            INPUT_DETAILS[0]["index"],
            input_data
        )

        interpreter.invoke()

        predictions = interpreter.get_tensor(
            OUTPUT_DETAILS[0]["index"]
        )[0]

        predicted_index = int(np.argmax(predictions))

        scores = {
            name: round(float(score) * 100, 2)
            for name, score in zip(CLASS_NAMES, predictions)
        }

        return jsonify({
            "success": True,
            "filename": filename,
            "model": "MobileNetV2 TFLite",
            "predicted_category": CLASS_NAMES[predicted_index],
            "model_score_percent": round(
                float(predictions[predicted_index]) * 100, 2
            ),
            "category_scores_percent": scores,
            "note": (
                "Experimental image-classification result, not a "
                "confirmed agricultural diagnosis. Consult a qualified "
                "agricultural expert before treatment."
            )
        })

    except (UnidentifiedImageError, OSError, ValueError):
        return jsonify({
            "success": False,
            "error": "Could not process this image."
        }), 400

    except Exception:
        app.logger.exception("Disease prediction failed")
        return jsonify({
            "success": False,
            "error": "Prediction service encountered an error."
        }), 500


@app.errorhandler(413)
def file_too_large(error):
    return jsonify({
        "success": False,
        "error": "Image exceeds the 10 MB limit."
    }), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
