import os
import cv2
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
from tqdm import tqdm

# ================= Configuration =================
# Set paths to point to your Celeb-DF-v2 dataset and a new processed folder
DATASET_ROOT = r"D:\Celeb-DF-v2"
OUTPUT_ROOT = r"D:\Celeb-DF-v2-processed"

# 6GB VRAM friendly settings: Extract fewer frames so sequence fits in memory
NUM_FRAMES = 15
FACE_SIZE = 299  # Xception model takes 299x299 images

# Dataset Limits to only train on 500 videos (keep it balanced: 250 Fake, 250 Real)
MAX_VIDEOS = {
    'Celeb-synthesis': 250, # Fake videos
    'Celeb-real': 125,      # Real videos
    'YouTube-real': 125     # Real videos
}

# Setup GPU for fast MTCNN face detection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Loaded MTCNN Face Detector on {device}")

# Initialize MTCNN (keep_all=False ensures we only take the main face, select_largest=True ignores background faces)
mtcnn = MTCNN(keep_all=False, select_largest=True, post_process=False, device=device)
# =================================================

def extract_and_crop(video_path, output_dir):
    """
    Extracts NUM_FRAMES uniformly from the video, targets the face, crops it out,
    and resizes it to 299x299 for our Xception-LSTM network.
    """
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) >= NUM_FRAMES:
        return # Skip if already processed

    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < NUM_FRAMES:
        cap.release()
        return

    # Uniformly space out the frames across the entire video
    frame_indices = np.linspace(0, total_frames - 1, NUM_FRAMES, dtype=int)
    
    saved_count = 0
    cap_idx = 0
    
    while cap.isOpened() and saved_count < NUM_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
            
        if cap_idx in frame_indices:
            # OpenCV loads as BGR, convert to RGB for MTCNN
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            
            # Detect and extract face
            try:
                # MTCNN gives us a PyTorch tensor shape [3, Height, Width]
                face = mtcnn(img) 
                if face is not None:
                    # Convert tensor back to image and resize to 299x299
                    face_array = face.permute(1, 2, 0).byte().cpu().numpy()
                    face_img = Image.fromarray(face_array)
                    face_img = face_img.resize((FACE_SIZE, FACE_SIZE))
                    
                    # Save local file
                    frame_path = os.path.join(output_dir, f"frame_{saved_count:02d}.jpg")
                    face_img.save(frame_path)
                    saved_count += 1
            except Exception as e:
                pass # If face is not found in this frame, just skip
                
        cap_idx += 1
        
    cap.release()

def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    # Common directories in Celeb-DF-v2
    # 'Celeb-synthesis' are the FAKE videos, 'Celeb-real' and 'YouTube-real' are the REAL videos.
    categories = ['Celeb-synthesis', 'Celeb-real', 'YouTube-real']
    
    for category in categories:
        cat_path = os.path.join(DATASET_ROOT, category)
        out_cat_path = os.path.join(OUTPUT_ROOT, category)
        
        if not os.path.exists(cat_path):
            print(f"Warning: Category folder not found {cat_path}")
            continue
            
        os.makedirs(out_cat_path, exist_ok=True)
        videos = [v for v in os.listdir(cat_path) if v.endswith('.mp4')]
        
        # Limit the number of videos processed for this category, so we only train on 500 total
        max_limit = MAX_VIDEOS.get(category, 0)
        videos = videos[:max_limit]
        
        print(f"Processing category: {category} ({len(videos)} videos)")
        
        # Use simple tqdm to track progress without breaking terminal output
        for video_name in tqdm(videos, desc=f"Cropping {category}"):
            video_path = os.path.join(cat_path, video_name)
            video_id = os.path.splitext(video_name)[0]
            output_dir = os.path.join(out_cat_path, video_id)
            
            extract_and_crop(video_path, output_dir)
            
    print(f"\nPhase 1 Complete! Cropped faces saved in {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()