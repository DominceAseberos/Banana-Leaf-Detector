import os
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.utils import resample
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV_PATH = os.path.join(BASE_DIR, "data.csv")
OUTPUT_JSON = os.path.join(BASE_DIR, "models", "knn_metrics.json")
OUTPUT_PNG = os.path.join(BASE_DIR, "models", "knn_metrics.png")

DISPLAY_NAMES = ["Unhealthy Leaf", "Healthy Leaf", "Non-Leaf"]
VAL_SPLIT = 0.2
RANDOM_SEED = 42


def evaluate():
    if not os.path.exists(CSV_PATH):
        print(f"CSV not found at {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} samples")

    df["label"] = df["label"].replace("Diseased leaf", "Unhealthy leaf")

    def normalize(l):
        l = l.lower()
        if "unhealthy" in l or "diseased" in l:
            return "Unhealthy Leaf"
        if "healthy" in l:
            return "Healthy Leaf"
        return "Non-Leaf"

    df["label"] = df["label"].apply(normalize)

    df_healthy = df[df["label"] == "Healthy Leaf"]
    df_unhealthy = df[df["label"] == "Unhealthy Leaf"]
    df_none = df[df["label"] == "Non-Leaf"]

    target_count = max(len(df_healthy), len(df_unhealthy))
    if len(df_none) > target_count:
        df_none = resample(
            df_none, replace=False, n_samples=target_count, random_state=RANDOM_SEED
        )

    df_balanced = pd.concat([df_healthy, df_unhealthy, df_none])
    df_balanced = df_balanced.sample(frac=1, random_state=RANDOM_SEED).reset_index(
        drop=True
    )

    print(f"Balanced: {len(df_balanced)} samples")
    for label in DISPLAY_NAMES:
        print(f"  {label}: {(df_balanced['label'] == label).sum()}")

    X = df_balanced.iloc[:, 2:].values.astype(float)
    y = df_balanced["label"].values

    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=VAL_SPLIT, random_state=RANDOM_SEED
    )
    train_idx, test_idx = next(sss.split(X, y))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    knn = KNeighborsClassifier(n_neighbors=5, weights="distance", p=2)
    knn.fit(X_train_scaled, y_train_enc)

    pred_enc = knn.predict(X_test_scaled)
    pred_labels = le.inverse_transform(pred_enc)

    acc = accuracy_score(y_test, pred_labels)
    cm = confusion_matrix(y_test, pred_labels, labels=DISPLAY_NAMES)
    report = classification_report(
        y_test,
        pred_labels,
        target_names=DISPLAY_NAMES,
        output_dict=True,
        zero_division=0,
    )

    print(f"\nAccuracy: {acc * 100:.2f}%")
    print()
    print(
        classification_report(
            y_test, pred_labels, target_names=DISPLAY_NAMES, zero_division=0
        )
    )

    # Save JSON
    metrics = {
        "val_accuracy": float(acc),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "class_names": DISPLAY_NAMES,
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved to {OUTPUT_JSON}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Oranges)
    ax.set_xticks(range(len(DISPLAY_NAMES)))
    ax.set_yticks(range(len(DISPLAY_NAMES)))
    ax.set_xticklabels(DISPLAY_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(DISPLAY_NAMES)
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("KNN Confusion Matrix")

    ax = axes[1]
    precisions = [report[n]["precision"] * 100 for n in DISPLAY_NAMES]
    recalls = [report[n]["recall"] * 100 for n in DISPLAY_NAMES]
    f1s = [report[n]["f1-score"] * 100 for n in DISPLAY_NAMES]
    x = np.arange(len(DISPLAY_NAMES))
    w = 0.25
    ax.bar(x - w, precisions, w, label="Precision", color="#e67e22")
    ax.bar(x, recalls, w, label="Recall", color="#3498db")
    ax.bar(x + w, f1s, w, label="F1-Score", color="#2ecc71")
    ax.set_xticks(x)
    ax.set_xticklabels(DISPLAY_NAMES, rotation=45, ha="right")
    ax.set_ylabel("Score (%)")
    ax.set_title(f"KNN Performance (Acc: {acc * 100:.2f}%)")
    ax.set_ylim(0, 100)
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {OUTPUT_PNG}")


if __name__ == "__main__":
    print("=" * 60)
    print("KNN Model Evaluation")
    print("=" * 60)
    evaluate()
