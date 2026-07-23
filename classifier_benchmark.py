import time
import numpy as np
import pandas as pd
import cv2
import faiss
import torch
from facenet_pytorch import InceptionResnetV1
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# ==========================================
# 0. Hardware Target Check
# ==========================================
print("=" * 50)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"HARDWARE TARGET: {device.type.upper()}")
if device.type == 'cuda':
    print(f"GPU DETECTED: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: GPU not detected by PyTorch.")
print("=" * 50)

# ==========================================
# 1. Dataset Preparation
# ==========================================
print("\nLoading Labeled Faces in the Wild (LFW) dataset...")
lfw = fetch_lfw_people(min_faces_per_person=40, resize=1.0)
X_raw, y = lfw.images, lfw.target

print("Resizing images and normalizing for PyTorch...")
X_rgb = []
for img in X_raw:
    img_uint8 = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    img_resized = cv2.resize(img_uint8, (160, 160))
    img_color = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
    X_rgb.append(img_color)

# Convert to PyTorch tensor format (Batch, Channels, Height, Width)
X_numpy = np.array(X_rgb, dtype=np.float32)
X_numpy = np.transpose(X_numpy, (0, 3, 1, 2))
X_numpy = (X_numpy - 127.5) / 128.0 
X_tensor = torch.tensor(X_numpy)

# ==========================================
# 2. Extracting Embeddings ON GPU
# ==========================================
print("\nLoading PyTorch FaceNet model to VRAM...")
embedder = InceptionResnetV1(pretrained='vggface2').eval().to(device)

print(f"Extracting 512D embeddings for {len(X_tensor)} faces...")
embeddings = []
BATCH_SIZE = 128 

start_time = time.time()
with torch.no_grad():
    for i in range(0, len(X_tensor), BATCH_SIZE):
        batch = X_tensor[i:i+BATCH_SIZE].to(device)
        emb = embedder(batch).cpu().numpy()
        embeddings.append(emb)

embeddings = np.vstack(embeddings)
print(f"Extraction complete in {time.time() - start_time:.2f} seconds.")

# L2 Normalization
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

X_train, X_test, y_train, y_test = train_test_split(embeddings, y, test_size=0.25, random_state=42)

# ==========================================
# 3. Model Benchmarking Setup (FIXED TIMING)
# ==========================================
results = []
test_size = len(X_test)

print("\nRunning Classifiers...")

# --- A. KNN ---
knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
t0 = time.time()
knn.fit(X_train, y_train)
train_time_knn = time.time() - t0

t1 = time.time()
y_pred_knn = knn.predict(X_test)
inf_time_knn = (time.time() - t1) / test_size * 1000

results.append({
    "Model": "KNN", 
    "Accuracy (%)": accuracy_score(y_test, y_pred_knn) * 100, 
    "Train/Index Time (s)": train_time_knn, 
    "Inference Time (ms/face)": inf_time_knn
})

# --- B. SVM ---
svm = SVC(kernel='linear', probability=True)
t0 = time.time()
svm.fit(X_train, y_train)
train_time_svm = time.time() - t0

t1 = time.time()
y_pred_svm = svm.predict(X_test)
inf_time_svm = (time.time() - t1) / test_size * 1000

results.append({
    "Model": "SVM", 
    "Accuracy (%)": accuracy_score(y_test, y_pred_svm) * 100, 
    "Train/Index Time (s)": train_time_svm, 
    "Inference Time (ms/face)": inf_time_svm
})

# --- C. MLP Network ---
mlp = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, random_state=42)
t0 = time.time()
mlp.fit(X_train, y_train)
train_time_mlp = time.time() - t0

t1 = time.time()
y_pred_mlp = mlp.predict(X_test)
inf_time_mlp = (time.time() - t1) / test_size * 1000

results.append({
    "Model": "MLP Network", 
    "Accuracy (%)": accuracy_score(y_test, y_pred_mlp) * 100, 
    "Train/Index Time (s)": train_time_mlp, 
    "Inference Time (ms/face)": inf_time_mlp
})

# --- D. FAISS ---
index = faiss.IndexFlatL2(512)
t0 = time.time()
index.add(np.array(X_train, dtype=np.float32))
train_time_faiss = time.time() - t0

t1 = time.time()
distances, indices = index.search(np.array(X_test, dtype=np.float32), 1)
y_pred_faiss = [y_train[idx[0]] for idx in indices]
inf_time_faiss = (time.time() - t1) / test_size * 1000

results.append({
    "Model": "FAISS", 
    "Accuracy (%)": accuracy_score(y_test, y_pred_faiss) * 100, 
    "Train/Index Time (s)": train_time_faiss, 
    "Inference Time (ms/face)": inf_time_faiss
})

# ==========================================
# 4. Results Export
# ==========================================
df = pd.DataFrame(results)

df["Accuracy (%)"] = df["Accuracy (%)"].round(2)
df["Train/Index Time (s)"] = df["Train/Index Time (s)"].round(4)
df["Inference Time (ms/face)"] = df["Inference Time (ms/face)"].round(4)

df.to_csv("recognition_benchmark_results.csv", index=False)

print("\n" + "=" * 75)
print(df.to_string(index=False))
print("=" * 75)