# Banana Leaf Disease Detector

A dual-model web application that detects **Healthy Leaf**, **Unhealthy Leaf**, and **Non-Leaf** images using both KNN and CNN (ResNet18) classifiers.

---

## Features

### Dual Model Architecture
- **KNN Scanner** — Existing K-Nearest Neighbors classifier (59 features: GLCM, LBP, HOG, color histograms)
- **CNN Scanner** — Deep learning classifier (ResNet18 transfer learning)

### Active Learning
- User feedback is logged locally (JSON) or to Firebase Firestore
- Model improvement data collected for future retraining

### Analytics Dashboard
- Scan history with thumbnails
- Disease distribution charts (Chart.js)
- Accuracy tracking per model

---

## Technology Stack

- **Backend:** Python 3.12, Flask
- **Database:** Firebase Firestore (optional) / Local JSON
- **ML (KNN):** scikit-learn, NumPy, Pandas, OpenCV, scikit-image
- **ML (CNN):** PyTorch, torchvision, ResNet18
- **Frontend:** HTML5, CSS3, JavaScript, Chart.js
- **Deployment:** Vercel / Render

---

## Project Structure

```
Banana-Leaf-Detector/
├── app.py                     # Flask application with dual routes
├── core/
│   ├── knn/
│   │   ├── extract_features.py   # 59-dim feature extraction
│   │   ├── knn_trainer.py        # KNN training script
│   │   └── __init__.py
│   ├── cnn/
│   │   ├── dataset_loader.py     # ImageFolder + transforms
│   │   ├── model.py              # ResNet18 wrapper
│   │   ├── cnn_trainer.py        # Training loop
│   │   ├── cnn_inference.py      # Single-image prediction
│   │   └── __init__.py
│   ├── feedback_store.py         # Local JSON feedback storage
│   └── __init__.py
├── models/                   # Saved model files
├── dataset/                  # Training & test images
├── static/                   # CSS, JS, uploads
├── templates/                # HTML views
├── data.csv                  # KNN feature dataset
└── requirements.txt          # Python dependencies
```

---

## Installation

```bash
git clone https://github.com/Domincee/Banana-Leaf-Detector.git
cd Banana-Leaf-Detector
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000` and navigate between the **KNN Scanner** and **CNN Scanner** tabs.

---

## Training the CNN Model

1. Download the training dataset from the [Google Drive link](https://drive.google.com/drive/folders/1mng06d0Y_U4hC7WM5hnbBNbuC5ohulcq)
2. Extract to `dataset/train_data/` with folders: `Healthy Leaf/`, `Diseased leaf/`, `None-leaf/`
3. Run the training script:
   ```bash
   python -m core.cnn.cnn_trainer
   ```
4. The trained model will be saved to `models/cnn_model.pth`

## KNN Model Training

```bash
python -m core.knn.knn_trainer
```

## License

© 2025 Domince Aseberos. MIT License.
