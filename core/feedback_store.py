import json
import os
from datetime import datetime

FEEDBACK_FILE = "feedback_local.json"


def _load():
    if not os.path.exists(FEEDBACK_FILE):
        return []
    with open(FEEDBACK_FILE, "r") as f:
        return json.load(f)


def _save(data):
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(data, f, indent=2)


def save_feedback(filename, prediction, is_correct, actual_label, model_type="knn"):
    data = _load()
    entry = {
        "timestamp": datetime.now().isoformat(),
        "filename": filename,
        "prediction": prediction,
        "is_correct": is_correct,
        "actual_label": actual_label,
        "model_type": model_type,
    }
    data.append(entry)
    _save(data)
    print(f"📝 Feedback saved locally: {model_type} -> {prediction}")
    return True


def get_feedback_history(limit=50, model_type=None):
    data = _load()
    if model_type:
        data = [d for d in data if d.get("model_type") == model_type]
    data.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    result = []
    for d in data[:limit]:
        is_correct = d.get("is_correct", False)
        actual = d.get("actual_label")
        pred = d.get("prediction")
        status = "unknown"
        if is_correct:
            status = "correct"
        elif actual and actual != pred:
            status = "corrected"
        else:
            status = "incorrect"
        result.append(
            {
                "timestamp": d.get("timestamp", ""),
                "filename": d.get("filename"),
                "prediction": pred,
                "actual": actual,
                "status": status,
            }
        )
    return result


def get_analytics_data(limit=500, model_type=None):
    data = _load()
    if model_type:
        data = [d for d in data if d.get("model_type") == model_type]

    total_scans = len(data)
    disease_counts = {}
    correct_count = 0
    incorrect_count = 0
    timeline = {}

    for d in data:
        actual = d.get("actual_label")
        pred = d.get("prediction")
        final_label = actual if actual and actual != "Unknown" else pred
        if final_label:
            disease_counts[final_label] = disease_counts.get(final_label, 0) + 1

        is_correct = d.get("is_correct", False)
        if is_correct:
            correct_count += 1
        elif actual and actual != pred:
            incorrect_count += 1

        ts = d.get("timestamp")
        if ts:
            date_str = ts[:10]
            timeline[date_str] = timeline.get(date_str, 0) + 1

    sorted_timeline = [{"date": k, "count": v} for k, v in sorted(timeline.items())]

    return {
        "total_scans": total_scans,
        "disease_counts": disease_counts,
        "accuracy": {"correct": correct_count, "incorrect": incorrect_count},
        "timeline": sorted_timeline,
    }
