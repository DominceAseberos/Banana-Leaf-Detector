import json
import os


def load_metrics(metrics_path=None):
    if metrics_path is None:
        metrics_path = os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ),
            "models",
            "cnn_metrics.json",
        )

    if not os.path.exists(metrics_path):
        print(f"Metrics not found at {metrics_path}.")
        print(
            "Run the Colab notebook to generate metrics, then download cnn_metrics.json to models/."
        )
        return None

    with open(metrics_path) as f:
        return json.load(f)


def display_metrics(metrics):
    if metrics is None:
        return

    history = {}
    for k in ("train_loss", "val_loss", "val_acc"):
        if k in metrics:
            history[k] = metrics[k]

    # Full metrics from new Colab notebook
    if "val_accuracy" in metrics:
        print(f"Validation Accuracy: {metrics['val_accuracy'] * 100:.2f}%")
        print()

        report = metrics["classification_report"]
        print("Classification Report:")
        print(
            f"{'':>16} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}"
        )
        for cls in report:
            if cls in ("accuracy", "macro avg", "weighted avg"):
                continue
            vals = report[cls]
            print(
                f"{cls:>16} {vals['precision']:>10.3f} {vals['recall']:>10.3f} {vals['f1-score']:>10.3f} {vals['support']:>10.0f}"
            )

        print()
        cm = metrics["confusion_matrix"]
        class_names = metrics["class_names"]
        print("Confusion Matrix:")
        header = f"{'':>16}" + "".join(f"{name:>16}" for name in class_names)
        print(header)
        for i, row in enumerate(cm):
            print(f"{class_names[i]:>16}" + "".join(f"{val:>16}" for val in row))

    # History-only (from old notebook run)
    elif history:
        print("Training History (full metrics require re-run on Colab):")
        best_acc = max(history["val_acc"])
        best_idx = history["val_acc"].index(best_acc)
        print(
            f"  Best Validation Accuracy: {best_acc * 100:.2f}% (epoch {best_idx + 1})"
        )
        print(f"  Final Train Loss: {history['train_loss'][-1]:.4f}")
        print(f"  Final Val Loss:   {history['val_loss'][-1]:.4f}")
        print()
        print("  Epoch  Train Loss  Val Loss    Val Acc")
        print("  " + "-" * 40)
        for i in range(len(history["train_loss"])):
            print(
                f"  {i + 1:>5}  {history['train_loss'][i]:.4f}     {history['val_loss'][i]:.4f}    {history['val_acc'][i] * 100:>5.2f}%"
            )


def metrics_image_path():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "models",
        "cnn_metrics.png",
    )


if __name__ == "__main__":
    print("=" * 60)
    print("CNN Model Evaluation")
    print("=" * 60)
    metrics = load_metrics()
    display_metrics(metrics)

    img_path = metrics_image_path()
    if os.path.exists(img_path):
        print(f"\nMetrics image: {img_path}")
    else:
        print(
            "\nMetrics image not found. Download cnn_metrics.png from Colab and place in models/."
        )
