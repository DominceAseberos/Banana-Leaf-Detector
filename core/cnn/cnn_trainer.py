import os
import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm
from .model import create_model


def train(model, train_loader, val_loader, epochs=15, lr=0.001, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr)

    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    model_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "models",
    )
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, "cnn_model.pth")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]", leave=False
        ):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(
                val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]", leave=False
            ):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct / total

        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch + 1}/{epochs} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  → Saved best model (val_acc={val_acc:.4f})")

    print(f"\nTraining complete. Best val_acc: {best_val_acc:.4f}")
    print(f"Model saved to: {save_path}")
    return history


if __name__ == "__main__":
    from core.cnn.dataset_loader import get_dataloaders

    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    possible_paths = [
        os.path.join(project_root, "raw_data"),
        os.path.join(project_root, "dataset", "raw_data"),
        os.path.join(project_root, "dataset", "train_data"),
    ]

    data_dir = None
    for p in possible_paths:
        if os.path.exists(p):
            data_dir = p
            break

    if data_dir is None:
        print(f"Training data not found. Tried:")
        for p in possible_paths:
            print(f"  - {p}")
        print(
            "Download from: https://drive.google.com/drive/folders/1mng06d0Y_U4hC7WM5hnbBNbuC5ohulcq"
        )
        print(
            "Extract to dataset/train_data/ and ensure subfolders: Diseased_Leaf/, Healthy_leaf/, Non_leaf/"
        )
        exit(1)

    print(f"Using data directory: {data_dir}")

    train_loader, val_loader, class_names = get_dataloaders(data_dir, batch_size=32)
    print(f"Classes: {class_names}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    model = create_model(num_classes=len(class_names))
    train(model, train_loader, val_loader, epochs=15)
