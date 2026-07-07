import cv2
import time
import pandas as pd
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from ultralytics import YOLO
from facenet_pytorch import MTCNN
import torch

# Configuration
VIDEO_PATH = "attendance_test_video.mp4"
NUM_FRAMES = 100

# 1. Initialize YOLOv8-Face (GPU)
yolo_model = YOLO("yolov8n-face.pt")
yolo_model.to('cuda')

# 2. Initialize MTCNN (GPU)
mtcnn_model = MTCNN(keep_all=True, device='cuda')

# 3. Initialize InsightFace / RetinaFace (GPU)
# 'buffalo_l' contains the RetinaFace detector + ArcFace recognition backbone
app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# Load Video
cap = cv2.VideoCapture(VIDEO_PATH)
frames = []
for _ in range(NUM_FRAMES):
    ret, frame = cap.read()
    if not ret: break
    frames.append(frame)
cap.release()

results = []

def run_benchmark(name, func, frame_list):
    start = time.time()
    for f in frame_list:
        func(f)
    fps = len(frame_list) / (time.time() - start)
    print(f"{name}: {fps:.2f} FPS")
    results.append({"Model": name, "FPS": fps})

# Benchmark YOLO
run_benchmark("YOLOv8-Face", lambda f: yolo_model(f, verbose=False), frames)

# Benchmark MTCNN
run_benchmark("MTCNN", lambda f: mtcnn_model.detect(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)), frames)

# Benchmark InsightFace (RetinaFace)
run_benchmark("InsightFace", lambda f: app.get(f), frames)

# Save results
pd.DataFrame(results).to_csv("gpu_benchmark_final.csv", index=False)