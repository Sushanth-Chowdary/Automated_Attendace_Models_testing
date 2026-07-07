import cv2
import torch
import numpy as np
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms

# ==========================================
# 1. SETUP MODELS
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the custom YOLOv8-Face model (Must be a model trained to output 5 facial keypoints)
# Replace with the path to your downloaded face-keypoint weights
yolo_model = YOLO("yolov8n-face.pt") 

# Load FaceNet (InceptionResnetV1)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Standard FaceNet image transforms (Resizing and Normalization)
trans = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ==========================================
# 2. THE ALIGNMENT MATHEMATICS
# ==========================================
def align_face(img, landmarks):
    """
    Rotates and scales the face so the eyes are perfectly level.
    landmarks: A list of 5 (x, y) coordinates [Left Eye, Right Eye, Nose, Mouth L, Mouth R]
    """
    left_eye = landmarks[0]
    right_eye = landmarks[1]
    
    # Calculate the angle between the eyes
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))
    
    # Center of the eyes
    eyes_center = (
        (left_eye[0] + right_eye[0]) // 2,
        (left_eye[1] + right_eye[1]) // 2
    )
    
    # Get the rotation matrix
    M = cv2.getRotationMatrix2D(eyes_center, angle, scale=1.0)
    
    # Apply the affine transformation to the entire image
    aligned_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC)
    
    return aligned_img

# ==========================================
# 3. THE PIPELINE EXECUTION
# ==========================================
def process_image(image_path):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print("Image not found.")
        return
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 1. Run YOLOv8-Face Detection
    results = yolo_model(img_rgb, verbose=False)
    
    embeddings_list = []
    
    # Loop through every detected face in the image
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy() # [x1, y1, x2, y2]
        
        # Check if this custom model outputs keypoints
        if not hasattr(result, 'keypoints') or result.keypoints is None:
            print("ERROR: This YOLO model does not output facial keypoints. Alignment is impossible.")
            return
            
        keypoints = result.keypoints.xy.cpu().numpy() # Shape: (N, 5, 2)
        
        for idx in range(len(boxes)):
            box = boxes[idx]
            landmarks = keypoints[idx]
            
            x1, y1, x2, y2 = map(int, box)
            
            # 2. Align the Face (CRITICAL FOR FACENET)
            # We align the whole image first before cropping to prevent cutting off corners
            aligned_img = align_face(img_rgb, landmarks)
            
            # 3. Crop the Face
            # Ensure coordinates are within image boundaries
            h, w, _ = aligned_img.shape
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            face_crop = aligned_img[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                continue
                
            # 4. Prepare for FaceNet (Resize to 160x160 and convert to Tensor)
            face_tensor = trans(face_crop).unsqueeze(0).to(device)
            
            # 5. Extract the Embedding!
            with torch.no_grad():
                embedding = resnet(face_tensor).cpu().numpy()[0]
                embeddings_list.append(embedding)
                
            print(f"Successfully extracted 512-d embedding for face {idx+1}")
            
    return embeddings_list

# Test the pipeline
# embeddings = process_image("test_image.jpg")