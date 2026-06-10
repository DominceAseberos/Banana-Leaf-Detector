import os
import json
import pickle
import uuid
from dotenv import load_dotenv

load_dotenv()
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.utils import resample
from core.knn.extract_features import extract_features
from werkzeug.utils import secure_filename
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
import jinja2
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from core.feedback_store import (
    save_feedback as save_feedback_local,
    get_feedback_history as get_feedback_history_local,
    get_analytics_data as get_analytics_data_local,
)

try:
    from core.cnn.cnn_inference import (
        load_model as load_cnn_model,
        predict as cnn_predict,
    )

    CNN_AVAILABLE = True
except Exception:
    CNN_AVAILABLE = False
    print("CNN module not available.")

app = FastAPI(title="Banana Leaf Detector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

jinja_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader("templates"),
    autoescape=True,
)


def render_template(name, **context):
    template = jinja_env.get_template(name)
    return HTMLResponse(template.render(context))


app.mount("/static", StaticFiles(directory="static"), name="static")

UPLOAD_FOLDER = "/tmp"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DATA_CSV = "data.csv"
FEATURE_CACHE = {}
MODEL_PATH = "models/knn_model.pkl"
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

RECOMMENDATIONS = {
    "Healthy Leaf": {
        "title": "Keep Up the Good Work!",
        "advice": "Your banana plant appears healthy. Continue regular care: adequate watering (avoid waterlogging), balanced fertilizer every 2-3 months, and maintain proper sunlight exposure.",
        "actions": [
            "Water regularly — keep soil moist but not waterlogged",
            "Apply balanced NPK fertilizer (e.g., 14-14-14) every 2-3 months",
            "Mulch around the base to retain moisture",
            "Monitor weekly for early signs of pests or disease",
        ],
    },
    "Unhealthy Leaf": {
        "title": "Action Needed — Disease Detected",
        "advice": "The leaf shows signs of disease. Common banana leaf diseases include Sigatoka leaf spot, Panama disease, or bacterial wilt. Early intervention is critical to prevent spread.",
        "actions": [
            "Prune and remove affected leaves immediately (sterilize tools)",
            "Apply fungicide (e.g., Mancozeb or Copper-based) if fungal spots are visible",
            "Improve air circulation — space plants adequately",
            "Avoid overhead watering to reduce leaf wetness",
            "Quarantine affected plant to prevent spread to neighboring plants",
            "Consult local agricultural extension office for lab diagnosis",
        ],
    },
    "Non-Leaf": {
        "title": "Not a Banana Leaf",
        "advice": "The uploaded image does not appear to be a banana leaf. This tool is designed to analyze banana leaf health.",
        "actions": [
            "Try uploading a clear photo of a banana leaf",
            "Ensure the leaf fills most of the frame",
            "Use good lighting without shadows",
        ],
    },
}


def get_recommendation(prediction):
    pred_lower = prediction.lower()
    if "unhealthy" in pred_lower:
        return RECOMMENDATIONS["Unhealthy Leaf"]
    if "healthy" in pred_lower:
        return RECOMMENDATIONS["Healthy Leaf"]
    return RECOMMENDATIONS["Non-Leaf"]


from PIL import Image


def _ensure_compat_format(file_path, ext):
    if ext.lower() == ".webp":
        png_path = file_path.rsplit(".", 1)[0] + ".png"
        Image.open(file_path).save(png_path, "PNG")
        os.remove(file_path)
        return png_path, ".png"
    return file_path, ext


# Load KNN model
try:
    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)
    knn = model_data["model"]
    scaler = model_data["scaler"]
    label_encoder_classes = model_data["classes"]
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(label_encoder_classes)
except Exception as e:
    print(f"Error loading KNN model: {e}")
    knn, scaler, label_encoder = None, None, None

# Load CNN model
cnn_model = None
if CNN_AVAILABLE:
    try:
        cnn_model = load_cnn_model()
        print("CNN model loaded successfully.")
    except Exception as e:
        print(f"CNN model not loaded: {e}")


def cnn_predict_safe(model, image_path):
    if model is None or not CNN_AVAILABLE:
        raise RuntimeError("CNN model is not available.")
    return cnn_predict(model, image_path)


def generate_explanation(probabilities, prediction):
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    top_label, top_score = sorted_probs[0]
    second_label, second_score = sorted_probs[1] if len(sorted_probs) > 1 else (None, 0)
    explanation = ""
    if top_score >= 90:
        explanation = (
            f"The model is highly confident ({top_score}%) that this is {top_label}."
        )
    elif top_score >= 70:
        explanation = f"The model is fairly confident ({top_score}%) in the {top_label} classification."
    elif top_score >= 50:
        explanation = f"The analysis suggests {top_label} ({top_score}%), but there is some ambiguity. The model also detected similarities to {second_label} ({second_score}%)."
    else:
        explanation = f"The result is uncertain. While {top_label} was the top match ({top_score}%), the features are very mixed."
    if "Unhealthy" in top_label:
        explanation += " Distinct discoloration or textural irregularities were detected on the leaf surface."
    elif "Healthy" in top_label and probabilities.get("Unhealthy", 0) > 20:
        explanation += " However, some small irregularities were noted, so keep an eye on the plant."
    elif "Non-Leaf" in top_label:
        explanation += " The image lacks the specific green/yellow color histograms and vein textures typically found in banana leaves."
    return explanation


# ============================
# Routes
# ============================
@app.get("/")
def index():
    return render_template("index.html")


@app.get("/scanner")
def scanner():
    return render_template("scanner.html")


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/upload")
async def upload_image(image: UploadFile = File(...)):
    filename = image.filename or "image.jpg"
    ext = os.path.splitext(secure_filename(filename))[1].lower() or ".jpg"
    if ext not in ALLOWED_EXTS:
        return JSONResponse({"error": f"Unsupported file type: {ext}"}, status_code=400)
    try:
        unique_filename = f"{uuid.uuid4().hex}{ext}"
        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        contents = await image.read()
        with open(file_path, "wb") as f:
            f.write(contents)
        file_path, ext = _ensure_compat_format(file_path, ext)
        unique_filename = os.path.basename(file_path)
        features = extract_features(file_path)
        FEATURE_CACHE[unique_filename] = features
        features_scaled = scaler.transform(features.reshape(1, -1))
        pred_encoded = knn.predict(features_scaled)[0]
        pred_label = label_encoder.inverse_transform([pred_encoded])[0]
        if "Diseased" in pred_label:
            pred_label = pred_label.replace("Diseased", "Unhealthy")
        prob_array = knn.predict_proba(features_scaled)[0]
        prob_class_labels = label_encoder.inverse_transform(knn.classes_)
        prob_dict = {}
        for cls, prob in zip(prob_class_labels, prob_array):
            cls_name = (
                cls.replace("Diseased", "Unhealthy") if "Diseased" in cls else cls
            )
            prob_dict[cls_name] = int(round(prob * 100))
        explanation = generate_explanation(prob_dict, pred_label)
        recommendation = get_recommendation(pred_label)
        return {
            "success": True,
            "prediction": pred_label,
            "probabilities": prob_dict,
            "explanation": explanation,
            "recommendation": recommendation,
            "filename": unique_filename,
        }
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.post("/upload/cnn")
async def upload_cnn(image: UploadFile = File(...)):
    filename = image.filename or "image.jpg"
    ext = os.path.splitext(secure_filename(filename))[1].lower() or ".jpg"
    if ext not in ALLOWED_EXTS:
        return JSONResponse({"error": f"Unsupported file type: {ext}"}, status_code=400)
    try:
        unique_filename = f"{uuid.uuid4().hex}{ext}"
        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        contents = await image.read()
        with open(file_path, "wb") as f:
            f.write(contents)
        file_path, ext = _ensure_compat_format(file_path, ext)
        unique_filename = os.path.basename(file_path)
        result = cnn_predict_safe(cnn_model, file_path)
        prediction = result["prediction"]
        prob_dict = result["probabilities"]
        confidence = result["confidence"]
        explanation_text = (
            f"The CNN model is {confidence}% confident this is {prediction}. "
            f"Probabilities: {', '.join(f'{k}: {v}%' for k, v in prob_dict.items())}"
        )
        recommendation = get_recommendation(prediction)
        return {
            "success": True,
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": prob_dict,
            "explanation": explanation_text,
            "recommendation": recommendation,
            "filename": unique_filename,
        }
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.get("/metrics")
def model_metrics():
    return render_template("model_metrics.html")


def _metrics_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


@app.get("/api/cnn-metrics")
def api_cnn_metrics():
    metrics_path = os.path.join(_metrics_dir(), "cnn_metrics.json")
    if not os.path.exists(metrics_path):
        return JSONResponse(
            {"success": False, "error": "Metrics not found."}, status_code=404
        )
    with open(metrics_path) as f:
        data = json.load(f)
    return {"success": True, "data": data}


@app.get("/api/cnn-metrics-image")
def api_cnn_metrics_image():
    img_path = os.path.join(_metrics_dir(), "cnn_metrics.jpg")
    if not os.path.exists(img_path):
        return JSONResponse(
            {"success": False, "error": "CNN metrics image not found."}, status_code=404
        )
    return FileResponse(img_path, media_type="image/jpeg")


@app.get("/api/knn-metrics")
def api_knn_metrics():
    metrics_path = os.path.join(_metrics_dir(), "knn_metrics.json")
    if not os.path.exists(metrics_path):
        return JSONResponse(
            {"success": False, "error": "KNN metrics not found."}, status_code=404
        )
    with open(metrics_path) as f:
        data = json.load(f)
    return {"success": True, "data": data}


@app.get("/api/knn-metrics-image")
def api_knn_metrics_image():
    img_path = os.path.join(_metrics_dir(), "knn_metrics.jpg")
    if not os.path.exists(img_path):
        return JSONResponse(
            {"success": False, "error": "KNN metrics image not found."}, status_code=404
        )
    return FileResponse(img_path, media_type="image/jpeg")


@app.get("/api/compare-metrics")
def api_compare_metrics():
    cnn_path = os.path.join(_metrics_dir(), "cnn_metrics.json")
    knn_path = os.path.join(_metrics_dir(), "knn_metrics.json")
    result = {"cnn": None, "knn": None}
    if os.path.exists(cnn_path):
        with open(cnn_path) as f:
            data = json.load(f)
            result["cnn"] = {
                "accuracy": data.get("val_accuracy", 0),
                "report": data.get("classification_report", {}),
                "class_names": data.get("class_names", []),
            }
    if os.path.exists(knn_path):
        with open(knn_path) as f:
            data = json.load(f)
            result["knn"] = {
                "accuracy": data.get("val_accuracy", 0),
                "report": data.get("classification_report", {}),
                "class_names": data.get("class_names", []),
            }
    return result


@app.get("/api/test-image/{folder}/{filename}")
def api_test_image(folder: str, filename: str):
    test_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "dataset", "test_data"
    )
    folder = os.path.basename(folder)
    filename = os.path.basename(filename)
    img_path = os.path.join(test_dir, folder, filename)
    if not os.path.exists(img_path):
        return JSONResponse(
            {"success": False, "error": "Image not found."}, status_code=404
        )
    return FileResponse(img_path)


@app.get("/api/compare-results")
def api_compare_results():
    results_path = os.path.join(_metrics_dir(), "comparison_results.json")
    if not os.path.exists(results_path):
        return JSONResponse(
            {"success": False, "error": "Comparison results not found."},
            status_code=404,
        )
    with open(results_path) as f:
        data = json.load(f)
    return {"success": True, "data": data}


@app.post("/feedback")
async def save_feedback(request: Request):
    try:
        data = await request.json()
        filename = data.get("filename", "unknown")
        prediction = data.get("prediction", "unknown")
        is_correct = data.get("correct", False)
        model_type = data.get("model_type", "knn")
        actual_label = data.get("actual_label")
        if not actual_label and is_correct:
            actual_label = prediction
        elif not actual_label:
            actual_label = "Unknown"
        save_feedback_local(filename, prediction, is_correct, actual_label, model_type)
        return {"success": True}
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.get("/api/analytics")
def analytics_api():
    try:
        data = get_analytics_data_local(limit=1000)
        return {"success": True, "data": data}
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.get("/history")
def get_history_route():
    history = get_feedback_history_local()
    return history


@app.post("/clear_history")
def clear_history():
    try:
        with open("feedback_local.json", "w") as f:
            json.dump([], f)
        return {"success": True}
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)
