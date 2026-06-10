import os
import json
import pickle
import glob
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TEST_DIR = os.path.join(BASE_DIR, "dataset", "test_data")
CNN_MODEL_PATH = os.path.join(BASE_DIR, "models", "cnn_model.pth")
KNN_MODEL_PATH = os.path.join(BASE_DIR, "models", "knn_model.pkl")
OUTPUT_JSON = os.path.join(BASE_DIR, "models", "comparison_results.json")
OUTPUT_IMAGE_DIR = os.path.join(BASE_DIR, "models", "comparison_samples")

LABEL_FOLDER_MAP = {
    "Diseased_leaf": "Unhealthy Leaf",
    "Healthy_leaf": "Healthy Leaf",
    "Non_leaf": "Non-Leaf",
}
DISPLAY_NAMES = ["Unhealthy Leaf", "Healthy Leaf", "Non-Leaf"]
DISPLAY_TO_FOLDER = {
    "Unhealthy Leaf": "Diseased_leaf",
    "Healthy Leaf": "Healthy_leaf",
    "Non-Leaf": "Non_leaf",
}


def load_knn():
    with open(KNN_MODEL_PATH, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["scaler"], data["classes"]


def knn_predict(knn, scaler, classes, image_path):
    from core.knn.extract_features import extract_features

    feats = extract_features(image_path).reshape(1, -1)
    feats_scaled = scaler.transform(feats)
    pred_enc = knn.predict(feats_scaled)[0]
    proba = knn.predict_proba(feats_scaled)[0]
    pred_label = classes[pred_enc]

    if "Diseased" in pred_label:
        pred_label = "Unhealthy Leaf"
    elif "None-leaf" in pred_label:
        pred_label = "Non-Leaf"
    elif "Unhealthy leaf" in pred_label:
        pred_label = "Unhealthy Leaf"

    prob_dict = {}
    for i, prob in enumerate(proba):
        cls = classes[i]
        cls = cls.replace("Diseased", "Unhealthy").replace("None-leaf", "Non-Leaf")
        if "Unhealthy leaf" in cls:
            cls = "Unhealthy Leaf"
        prob_dict[cls] = round(prob * 100, 2)

    confidence = round(max(proba) * 100, 2)
    return pred_label, confidence, prob_dict


def normalize(label):
    l = label.lower()
    if "unhealthy" in l or "diseased" in l:
        return "Unhealthy Leaf"
    if "healthy" in l:
        return "Healthy Leaf"
    return "Non-Leaf"


def main():
    print("Loading CNN model...")
    from core.cnn.cnn_inference import load_model as load_cnn, predict as cnn_predict

    cnn = load_cnn(CNN_MODEL_PATH)
    print("Loading KNN model...")
    knn, scaler, classes = load_knn()

    os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)

    image_paths = []
    for folder in sorted(os.listdir(TEST_DIR)):
        folder_path = os.path.join(TEST_DIR, folder)
        if not os.path.isdir(folder_path):
            continue
        for fname in sorted(os.listdir(folder_path)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                image_paths.append((os.path.join(folder_path, fname), folder))

    results = []
    cnn_correct_knns_wrong = []
    knn_correct_cnn_wrong = []

    print(f"Testing {len(image_paths)} images...")
    for path, folder in image_paths:
        true_label = LABEL_FOLDER_MAP.get(folder, folder)

        cnn_result = cnn_predict(cnn, path)
        cnn_pred = cnn_result["prediction"]
        cnn_conf = cnn_result["confidence"]

        knn_pred, knn_conf, knn_probs = knn_predict(knn, scaler, classes, path)

        cnn_correct = normalize(cnn_pred) == normalize(true_label)
        knn_correct = normalize(knn_pred) == normalize(true_label)

        fname = os.path.basename(path)
        entry = {
            "filename": fname,
            "folder": folder,
            "true_label": true_label,
            "cnn_prediction": cnn_pred,
            "cnn_confidence": cnn_conf,
            "cnn_correct": cnn_correct,
            "knn_prediction": knn_pred,
            "knn_confidence": knn_conf,
            "knn_correct": knn_correct,
        }
        results.append(entry)

        if cnn_correct and not knn_correct:
            cnn_correct_knns_wrong.append(entry)
        if knn_correct and not cnn_correct:
            knn_correct_cnn_wrong.append(entry)

    print(f"\nTotal: {len(results)} images")
    print(f"CNN correct, KNN wrong: {len(cnn_correct_knns_wrong)}")
    print(f"KNN correct, CNN wrong: {len(knn_correct_cnn_wrong)}")
    print(
        f"Both correct: {sum(1 for r in results if r['cnn_correct'] and r['knn_correct'])}"
    )
    print(
        f"Both wrong: {sum(1 for r in results if not r['cnn_correct'] and not r['knn_correct'])}"
    )
    print(f"CNN overall: {sum(1 for r in results if r['cnn_correct'])}/{len(results)}")
    print(f"KNN overall: {sum(1 for r in results if r['knn_correct'])}/{len(results)}")

    output = {
        "total": len(results),
        "cnn_wins": len(cnn_correct_knns_wrong),
        "knn_wins": len(knn_correct_cnn_wrong),
        "both_correct": sum(
            1 for r in results if r["cnn_correct"] and r["knn_correct"]
        ),
        "both_wrong": sum(
            1 for r in results if not r["cnn_correct"] and not r["knn_correct"]
        ),
        "cnn_accuracy": round(
            sum(1 for r in results if r["cnn_correct"]) / len(results) * 100, 2
        ),
        "knn_accuracy": round(
            sum(1 for r in results if r["knn_correct"]) / len(results) * 100, 2
        ),
        "cnn_wins_samples": cnn_correct_knns_wrong[:8],
        "knn_wins_samples": knn_correct_cnn_wrong[:8],
        "all_results": results,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {OUTPUT_JSON}")


if __name__ == "__main__":
    print("=" * 60)
    print("Model Comparison: CNN vs KNN")
    print("=" * 60)
    main()
