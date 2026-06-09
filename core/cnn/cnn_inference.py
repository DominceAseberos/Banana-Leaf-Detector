import os
import torch
from PIL import Image
from torchvision import transforms
from .model import create_model
from .dataset_loader import OUTPUT_MAP, val_transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path=None):
    if model_path is None:
        model_dir = os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ),
            "models",
        )
        model_path = os.path.join(model_dir, "cnn_model.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"CNN model not found at {model_path}. "
            "Train first: python -m core.cnn.cnn_trainer"
        )

    model = create_model(num_classes=3, pretrained=False)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    tensor = val_transform(image).unsqueeze(0)
    return tensor.to(device)


def predict(model, image_path):
    tensor = preprocess_image(image_path)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        confidence, predicted = torch.max(probs, 1)

    class_idx = predicted.item()

    return {
        "prediction": OUTPUT_MAP[class_idx],
        "confidence": round(confidence.item() * 100, 2),
        "probabilities": {
            OUTPUT_MAP[i]: round(probs[0][i].item() * 100, 2) for i in range(3)
        },
    }
