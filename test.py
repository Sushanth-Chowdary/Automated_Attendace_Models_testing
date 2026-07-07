import os
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score
import hdbscan

# 1. Initialize Models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on device: {device}")

# MTCNN detects faces and crops them to 160x160 (required by FaceNet)
mtcnn = MTCNN(image_size=160, margin=0, keep_all=False, device=device)

# InceptionResnetV1 pretrained on vggface2 outputs 512-dimensional embeddings
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def extract_embeddings_from_dataset(dataset_path):
    embeddings = []
    true_labels = []
    label_to_name = {}
    
    current_label_id = 0
    
    # Loop through each person's folder
    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_dir): continue
        
        label_to_name[current_label_id] = person_name
        
        for image_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, image_name)
            try:
                img = Image.open(img_path).convert('RGB')
                
                # Detect and crop the face using MTCNN
                face_tensor = mtcnn(img)
                
                if face_tensor is not None:
                    # MTCNN outputs a tensor of shape (3, 160, 160)
                    # Resnet expects a batch dimension: (1, 3, 160, 160)
                    face_batch = face_tensor.unsqueeze(0).to(device)
                    
                    # Generate 512-d embedding
                    with torch.no_grad():
                        embedding = resnet(face_batch).cpu().numpy()[0]
                        
                    embeddings.append(embedding)
                    true_labels.append(current_label_id)
            except Exception as e:
                print(f"Could not process {img_path}: {e}")
                
        current_label_id += 1
        
    return np.array(embeddings), np.array(true_labels), label_to_name

# 2. Process Dataset
# Replace 'path/to/your/dataset' with the actual folder path
DATASET_PATH = "path/to/your/dataset" 

print("Processing images and extracting 512-d embeddings...")
X, true_labels, label_map = extract_embeddings_from_dataset(DATASET_PATH)

print(f"Extracted {len(X)} faces across {len(label_map)} people.")

if len(X) > 0:
    # 3. L2 Normalize Embeddings (Critical for FaceNet cosine similarity)
    X_normalized = normalize(X, norm='l2')

    # 4. Initialize Clustering Models
    num_people = len(label_map)
    
    models = {
        "K-Means": KMeans(n_clusters=num_people, random_state=42, n_init='auto'),
        # Note: eps tuned for 512-d normalized space; you may need to adjust this (0.4 to 0.8)
        "DBSCAN": DBSCAN(eps=0.6, min_samples=3), 
        "HDBSCAN": hdbscan.HDBSCAN(min_cluster_size=3, metric='euclidean')
    }

    # 5. Evaluate
    print("\n--- Clustering Evaluation Results ---")
    for name, model in models.items():
        predicted_labels = model.fit_predict(X_normalized)
        
        ari = adjusted_rand_score(true_labels, predicted_labels)
        
        unique_labels = set(predicted_labels)
        n_clusters_found = len(unique_labels) - (1 if -1 in unique_labels else 0)
        noise_points = list(predicted_labels).count(-1)
        
        mask = predicted_labels != -1
        if n_clusters_found > 1 and len(X_normalized[mask]) > 0:
            sil = silhouette_score(X_normalized[mask], predicted_labels[mask])
        else:
            sil = -1.0 
            
        print(f"\nModel: {name}")
        print(f" - Clusters Found: {n_clusters_found} (True count: {num_people})")
        print(f" - Noise Points:   {noise_points} out of {len(X)}")
        print(f" - ARI:            {ari:.4f} (Closer to 1.0 is better)")
        print(f" - Silhouette:     {sil:.4f} (Closer to 1.0 is better)")
else:
    print("No faces were detected in the dataset. Check your image paths.")