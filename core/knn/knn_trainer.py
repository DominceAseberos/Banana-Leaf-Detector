import os
import pickle
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample

# ============================
# Load Feature CSV
# ============================
csv_path = "data.csv"
if not os.path.exists(csv_path):
    print("❌ CSV file not found.")
    exit()

df = pd.read_csv(csv_path)
print(f"✅ Loaded {len(df)} samples")

df = df.drop(columns=["path"])

# =====replace 'Diseased leaf' with 'Unhealthy leaf'=====
df["label"] = df["label"].replace("Diseased leaf", "Unhealthy leaf")

# ============================
# Separate by class for balancing
# ============================
df_healthy = df[df["label"] == "Healthy Leaf"]
df_none = df[df["label"] == "None-leaf"]
df_unhealthy = df[df["label"] == "Unhealthy leaf"]

# Initialize label encoder
le_encoder = LabelEncoder()

# ============================
# Downsample majority class (None-leaf)
# ============================
df_none_down = resample(
    df_none,
    replace=False,  # no replacement
    n_samples=len(df_healthy),  # match minority class
    random_state=42,
)  # reproducibility

# Combine all classes back
df_balanced = pd.concat([df_healthy, df_unhealthy, df_none_down])

# ============================
# Prepare Features and Labels
# ============================
X_bal = df_balanced.drop("label", axis=1).values
y_bal = le_encoder.fit_transform(df_balanced["label"])
class_names = le_encoder.classes_  # Needed for reports & confusion matrix

# ============================
# Train-Test Split (80-20)
# ============================
X_train, X_test, y_train, y_test = train_test_split(
    X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal
)
print(f"📊 Train: {len(X_train)}, Test: {len(X_test)}")

# ============================
# Normalize Features
# ============================
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ============================
# Hyperparameter Grid for KNN
# ============================
param_grid = {
    "n_neighbors": [5, 7, 9, 11],  # Number of neighbors
    "weights": ["uniform", "distance"],  # Weight function
    "p": [1, 2],  # Distance metric: 1=Manhattan, 2=Euclidean
}

# ============================
# Train KNN with GridSearchCV
# ============================
knn_model = KNeighborsClassifier()
grid = GridSearchCV(knn_model, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid.fit(X_train, y_train)

# Best KNN model
model = grid.best_estimator_
print(f"Best Params: {grid.best_params_}")

# ============================
# Evaluate Model
# ============================
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Test Accuracy: {acc * 100:.2f}%")

# Classification report
print(
    "\nClassification Report:\n",
    classification_report(y_test, y_pred, target_names=class_names),
)

# ============================
# Confusion Matrix
# ============================
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.close()
print("📊 Confusion matrix saved as 'confusion_matrix.png'")

# ============================
# Save Model and Scaler
# ============================
model_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
model_path = os.path.join(model_dir, "models", "knn_model.pkl")
os.makedirs(os.path.dirname(model_path), exist_ok=True)
with open(model_path, "wb") as f:
    pickle.dump({"model": model, "scaler": scaler, "classes": class_names}, f)
print(f"💾 Model saved as '{model_path}'")
