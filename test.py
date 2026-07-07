import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
import kagglehub
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
import hdbscan

# ==========================================
# 1. SETUP & DATASET DOWNLOAD
# ==========================================
# Your specific SMB network share path (ONLY used for saving the CSV results)
BASE_DIR = "/run/user/1000/gvfs/smb-share:server=10.23.20.56,share=rkgnas_user2/EE23B044_testing"
RESULTS_FILE = os.path.join(BASE_DIR, "clustering_benchmark_results.csv")

print("--- Phase 1: Dataset Acquisition ---")
print("Downloading/Locating LFW dataset in local cache...")
cached_path = kagglehub.dataset_download("jessicali9530/lfw-dataset")

# The Kaggle LFW dataset nests the images deep inside these folders
DATASET_DIR = os.path.join(cached_path, "lfw-deepfunneled", "lfw-deepfunneled")
print(f"Reading images directly from local cache: {DATASET_DIR}")

# ==========================================
# 2. MODEL INITIALIZATION
# ==========================================
print("\n--- Phase 2: Model Initialization ---")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Active Device: {device}")
if device.type == 'cuda':
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

mtcnn = MTCNN(image_size=160, margin=0, keep_all=False, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# ==========================================
# 3. EMBEDDING EXTRACTION
# ==========================================
print("\n--- Phase 3: Extracting 512-d Embeddings ---")
embeddings = []
true_labels = []
label_map = {}
current_label_id = 0

MIN_FACES_PER_PERSON = 10 

for person_name in os.listdir(DATASET_DIR):
    person_dir = os.path.join(DATASET_DIR, person_name)
    if not os.path.isdir(person_dir): continue
    
    images = [f for f in os.listdir(person_dir) if f.endswith(('.jpg', '.png'))]
    if len(images) < MIN_FACES_PER_PERSON:
        continue
        
    label_map[current_label_id] = person_name
    
    for image_name in images:
        img_path = os.path.join(person_dir, image_name)
        try:
            img = Image.open(img_path).convert('RGB')
            face_tensor = mtcnn(img) 
            
            if face_tensor is not None:
                face_batch = face_tensor.unsqueeze(0).to(device)
                with torch.no_grad():
                    embedding = resnet(face_batch).cpu().numpy()[0]
                    
                embeddings.append(embedding)
                true_labels.append(current_label_id)
        except Exception as e:
            pass 
            
    current_label_id += 1
    # Limiting to 20 students for the benchmark speed
    if current_label_id >= 20: 
        break

X = np.array(embeddings)
true_labels = np.array(true_labels)
num_known_students = len(np.unique(true_labels))

print(f"Extracted {len(X)} valid faces across {num_known_students} people.")

X_normalized = normalize(X, norm='l2')

# ==========================================
# 4. CLUSTERING BENCHMARK
# ==========================================
print("\n--- Phase 4: Clustering Benchmark ---")

models = {
    "K-Means": KMeans(n_clusters=num_known_students, random_state=42, n_init='auto'),
    "DBSCAN": DBSCAN(eps=0.7, min_samples=3),
    "HDBSCAN": hdbscan.HDBSCAN(min_cluster_size=3, metric='euclidean'),
    "Agglomerative": AgglomerativeClustering(n_clusters=None, distance_threshold=0.8, metric='euclidean', linkage='average')
}

results_data = []

for name, model in models.items():
    predicted_labels = model.fit_predict(X_normalized)
    
    ari = adjusted_rand_score(true_labels, predicted_labels)
    
    unique_labels = set(predicted_labels)
    noise_points = list(predicted_labels).count(-1)
    n_clusters_found = len(unique_labels) - (1 if -1 in unique_labels else 0)
    
    mask = predicted_labels != -1
    if n_clusters_found > 1 and len(X_normalized[mask]) > 0:
        sil = silhouette_score(X_normalized[mask], predicted_labels[mask])
    else:
        sil = -1.0
        
    print(f"[{name}]")
    print(f"  Clusters: {n_clusters_found}/{num_known_students} | Noise: {noise_points} | ARI: {ari:.3f} | Silhouette: {sil:.3f}")
    
    results_data.append({
        "Model": name,
        "Clusters_Found": n_clusters_found,
        "Target_Clusters": num_known_students,
        "Noise_Faces": noise_points,
        "ARI_Score": round(ari, 4),
        "Silhouette_Score": round(sil, 4)
    })

# ==========================================
# 5. EXPORT RESULTS
# ==========================================
print("\n--- Phase 5: Exporting Results ---")
df = pd.DataFrame(results_data)
# Ensure the network directory exists before writing
os.makedirs(BASE_DIR, exist_ok=True)
df.to_csv(RESULTS_FILE, index=False)
print(f"Benchmark complete! Results saved to: {RESULTS_FILE}")