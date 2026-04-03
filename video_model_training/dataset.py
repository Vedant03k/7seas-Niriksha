import os
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image

class VideoDeepfakeDataset(Dataset):
    def __init__(self, data_dir, sequence_length=15, transform=None):
        """
        Custom PyTorch Dataset to load video sequences.
        Reads the cropped faces from Phase 1 and groups them into sequences.
        """
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.transform = transform
        self.samples = []
        
        print(f"Loading dataset from: {data_dir}")
        # Prepare dataset: 'Celeb-synthesis' -> 1 (FAKE), Others -> 0 (REAL)
        categories = ['Celeb-synthesis', 'Celeb-real', 'YouTube-real']
        for category in categories:
            label = 1.0 if category == 'Celeb-synthesis' else 0.0
            cat_path = os.path.join(data_dir, category)
            if not os.path.exists(cat_path):
                continue
                
            video_folders = os.listdir(cat_path)
            for folder in video_folders:
                folder_path = os.path.join(cat_path, folder)
                if os.path.isdir(folder_path):
                    frames = sorted(glob.glob(os.path.join(folder_path, "*.jpg")))
                    if len(frames) >= 5:  # ensure we have a usable amount of frames
                        self.samples.append((folder_path, label, frames))
                        
        print(f"Successfully tracked {len(self.samples)} video sequences.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        folder_path, label, frame_paths = self.samples[idx]
        
        # Select frames (pad if less than sequence_length, or truncate)
        if len(frame_paths) > self.sequence_length:
            # take evenly spaced frames
            indices = torch.linspace(0, len(frame_paths)-1, self.sequence_length).long()
            selected_frames = [frame_paths[i] for i in indices]
        else:
            # pad by repeating the last frame if we fall short
            selected_frames = frame_paths + [frame_paths[-1]] * (self.sequence_length - len(frame_paths))

        frames = []
        for img_path in selected_frames[:self.sequence_length]:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            frames.append(img)
            
        # Stack into tensor: (sequence_length, channels, height, width)
        sequence_tensor = torch.stack(frames)
        
        # Return as (Features, Label)
        return sequence_tensor, torch.tensor([label], dtype=torch.float32)
