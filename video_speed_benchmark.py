import os
import time
import cv2
import urllib.request
import pandas as pd
from mtcnn import MTCNN
from retinaface import RetinaFace
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# ==========================================
# Configurations
# ==========================================
# Standard face detection test video from Intel IoT DevKit
VIDEO_URL = "https://github.com/intel-iot-devkit/sample-videos/blob/master/head-pose-face-detection-female-and-male.mp4?raw=true"
VIDEO_PATH = "standard_test_video.mp4"
MAX_FRAMES = 150  # ~5 seconds of video (keeps CPU testing reasonable)
CSV_FILENAME = "video_detector_benchmark_results.csv"

# ==========================================
# Download Standard Video
# ==========================================
if not os.path.exists(VIDEO_PATH):
    print("Downloading standard test video...")
    urllib.request.urlretrieve(VIDEO_URL, VIDEO_PATH)
    print("Download complete!\n")

# ==========================================
# Model Initialization
# ==========================================
print("Loading Models...")

# 1. MTCNN
mtcnn_model = MTCNN()

# 2. RetinaFace (Direct class calls)
# 3. YOLOv8-Face
print("Loading YOLOv8-Face weights...")
yolo_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
yolo_model = YOLO(yolo_path)

# ==========================================
# Video Verification
# ==========================================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Could not open video file at {VIDEO_PATH}.")

total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Successfully loaded video: {frame_width}x{frame_height} resolution, {total_video_frames} total frames.")

benchmark_frames = min(MAX_FRAMES, total_video_frames)
print(f"Running benchmark over {benchmark_frames} frames...\n")

# ==========================================
# Benchmarking Setup
# ==========================================
results = {
    "MTCNN": {"time": 0.0, "faces_detected": 0},
    "RetinaFace": {"time": 0.0, "faces_detected": 0},
    "YOLOv8": {"time": 0.0, "faces_detected": 0}
}

frame_count = 0

while cap.isOpened() and frame_count < benchmark_frames:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame_count += 1
    if frame_count % 15 == 0:
        print(f"Processing frame {frame_count}/{benchmark_frames}...")
        
    # Standardize color spacing (OpenCV uses BGR, models expect RGB)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # --- 1. MTCNN ---
    start_time = time.time()
    try:
        mt_res = mtcnn_model.detect_faces(img_rgb)
        results["MTCNN"]["time"] += (time.time() - start_time)
        results["MTCNN"]["faces_detected"] += len(mt_res)
    except Exception:
        results["MTCNN"]["time"] += (time.time() - start_time)

    # --- 2. RetinaFace ---
    start_time = time.time()
    try:
        rf_res = RetinaFace.detect_faces(img_rgb)
        results["RetinaFace"]["time"] += (time.time() - start_time)
        if isinstance(rf_res, dict):
            results["RetinaFace"]["faces_detected"] += len(rf_res)
    except Exception:
        results["RetinaFace"]["time"] += (time.time() - start_time)

    # --- 3. YOLOv8 ---
    start_time = time.time()
    try:
        yolo_res = yolo_model(img_rgb, verbose=False)
        results["YOLOv8"]["time"] += (time.time() - start_time)
        for r in yolo_res:
            results["YOLOv8"]["faces_detected"] += len(r.boxes)
    except Exception:
        results["YOLOv8"]["time"] += (time.time() - start_time)

cap.release()

# ==========================================
# Calculate Metrics & Export
# ==========================================
print("\nCompiling performance data...")
final_data = []

# Using RetinaFace as the high-accuracy baseline to calculate a "Relative Capture Rate"
baseline_faces = results["RetinaFace"]["faces_detected"] if results["RetinaFace"]["faces_detected"] > 0 else 1

for model, data in results.items():
    avg_fps = frame_count / data["time"] if data["time"] > 0 else 0
    relative_capture = (data["faces_detected"] / baseline_faces) * 100
    
    final_data.append({
        "Model": model,
        "Avg FPS": round(avg_fps, 2),
        "Total Faces Found": data["faces_detected"],
        "Relative Capture Rate (%)": round(relative_capture, 2),
        "Total Processing Time (s)": round(data["time"], 2)
    })

df = pd.DataFrame(final_data)
df.to_csv(CSV_FILENAME, index=False)

print(f"\nBenchmarking complete! Metrics saved to: {CSV_FILENAME}")
print("=" * 65)
print(df.to_string(index=False))
print("=" * 65)