import torch
from PIL import Image
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm
import joblib
from transformers import CLIPProcessor, CLIPModel
import warnings
warnings.filterwarnings("ignore")


# Load CLIP Model

print("Loading CLIP model (OpenAI)...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()


# Dataset Setup

dataset_path = "dataset"
labels = sorted(os.listdir(dataset_path))
X, y = [], []

print(f"\n Found {len(labels)} classes: {labels}\n")
print(" Extracting CLIP embeddings (Original + Augmented)...\n")

#  Feature Extraction

for label in tqdm(labels, desc="Processing classes"):
    folder = os.path.join(dataset_path, label)
    if not os.path.isdir(folder):
        continue

    # Include both main folder and 'augmented' subfolder
    for subfolder in ["", "augmented"]:
        path_to_use = os.path.join(folder, subfolder)
        if not os.path.exists(path_to_use):
            continue

        for file in tqdm(os.listdir(path_to_use), desc=f"{label}", leave=False):
            if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            img_path = os.path.join(path_to_use, file)
            try:
                img = Image.open(img_path).convert("RGB")
                inputs = processor(images=img, return_tensors="pt").to(device)
                with torch.no_grad():
                    embedding = model.get_image_features(**inputs)
                X.append(embedding.squeeze().cpu().numpy())
                y.append(label)
            except Exception as e:
                print(f" Skipped {img_path}: {e}")

X = np.array(X)
y = np.array(y)
print(f"\n Extracted {len(X)} embeddings successfully.\n")


# Normalize the feature vectors

print(" Normalizing features...")
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Save the scaler for future predictions
joblib.dump(scaler, "scaler.pkl")


# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f" Train samples: {len(X_train)}, Test samples: {len(X_test)}\n")


# Train Multiple Classifiers

classifiers = {
    "SVM (RBF)": SVC(kernel='rbf', probability=True, C=3, gamma='scale',
                     class_weight='balanced', random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "MLP (Neural Net)": MLPClassifier(hidden_layer_sizes=(256, 128),
                                      activation='relu', max_iter=300, random_state=42)
}

results = {}
best_model = None
best_acc = 0

for name, clf in classifiers.items():
    print(f" Training {name}...")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f" {name} Accuracy: {acc*100:.2f}%")
    if acc > best_acc:
        best_acc = acc
        best_model = clf


# Model Comparison

print("\n Model Comparison:")
for name, acc in results.items():
    print(f"{name:<20} : {acc*100:.2f}%")

print(f"\n Best Model: {type(best_model).__name__} with Accuracy: {best_acc*100:.2f}%\n")

# Evaluate the Best Model

y_pred = best_model.predict(X_test)
print("Classification Report:\n")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save Model Bundle

joblib.dump({
    "model": best_model,
    "scaler": scaler,
    "labels": labels
}, "sign_model_bundle.pkl")

print("\n Model and scaler saved to 'sign_model_bundle.pkl'")


# PCA Visualization

print("\n Generating PCA visualization...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
plt.figure(figsize=(8, 6))
for label in np.unique(y):
    idx = y == label
    plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=label, alpha=0.6)
plt.title("CLIP Feature Clusters by Gesture")
plt.legend()
plt.show()

print("\n All done! Your model is ready for real-time sign detection.")
