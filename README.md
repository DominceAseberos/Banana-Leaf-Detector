---
title: Banana Leaf Detector
emoji: 🍌
colorFrom: indigo
colorTo: green
sdk: docker
pinned: false
---

# Banana Leaf Disease Detector

A dual-model web application that detects **Healthy Leaf**, **Unhealthy Leaf**, and **Non-Leaf** images using both KNN and CNN (ResNet18) classifiers.

- **Website:** [banana-leaf-detector.vercel.app](https://banana-leaf-detector.vercel.app)
- **GitHub:** [github.com/Domincee/Banana-Leaf-Detector](https://github.com/Domincee/Banana-Leaf-Detector)

## Features

### Dual Model Architecture
- **KNN Scanner** — K-Nearest Neighbors classifier (59 features: GLCM, LBP, HOG, color histograms)
- **CNN Scanner** — Deep learning classifier (ResNet18 transfer learning, ~95% validation accuracy)

### Model Comparison Dashboard
Per-class precision/recall/F1 comparison table with "Where Models Disagree" gallery showing test images where CNN and KNN differ.

### Active Learning
User feedback is logged locally for future retraining.

## Technology Stack

- **Backend:** Python 3.12, FastAPI, Uvicorn
- **ML (KNN):** scikit-learn, NumPy, Pandas, OpenCV, scikit-image
- **ML (CNN):** PyTorch, torchvision, ResNet18
- **Frontend:** HTML5, CSS3, JavaScript
- **Deployment:** Docker (HF Spaces)

## Project Structure

```
├── app.py                     # FastAPI application
├── core/
│   ├── knn/                   # KNN feature extraction, training, evaluation
│   ├── cnn/                   # CNN model, training, inference, evaluation
│   └── compare_models.py      # CNN vs KNN comparison on test_data
├── models/                    # Saved model files + metrics
├── dataset/test_data/         # 39 test images (13 per class)
├── static/                    # CSS, favicon
├── templates/                 # HTML views (index, scanner, metrics)
├── data.csv                   # KNN feature dataset (2878 samples)
├── Dockerfile                 # HF Spaces deployment
└── requirements.txt           # Python dependencies
```

## Model Performance

| Metric | CNN (ResNet18) | KNN |
|---|---|---|
| Validation/Test Accuracy | 95.2% | 88.3% |
| Test Data (39 images) | 100% | 74% |

## Training the CNN Model

Use the Colab notebook (`colab_train_cnn.ipynb`) to train on GPU. Download the trained `.pth`, `.json`, and `.png` files to `models/`.
