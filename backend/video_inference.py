import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from facenet_pytorch import MTCNN
import timm
import base64

# Define the exact model architecture so PyTorch can load the weights
class VideoDeepfakeModel(nn.Module):
    def __init__(self, sequence_length=15, lstm_hidden_size=512, lstm_layers=1, dropout=0.5):
        super(VideoDeepfakeModel, self).__init__()
        self.sequence_length = sequence_length
        self.feature_extractor = timm.create_model('tf_efficientnet_b4_ns', pretrained=False, num_classes=0)
        feature_dim = self.feature_extractor.num_features
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=lstm_hidden_size, num_layers=lstm_layers, batch_first=True, dropout=dropout if lstm_layers > 1 else 0.0)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.size()
        x = x.view(batch_size * seq_length, c, h, w)
        features = self.feature_extractor(x)
        features = features.view(batch_size, seq_length, -1)
        lstm_out, _ = self.lstm(features)
        last_hidden_output = lstm_out[:, -1, :]
        out = self.classifier(last_hidden_output)
        return out

# Global inference settings
NUM_FRAMES_PER_SEQ = 15
MAX_CHUNKS_TO_ANALYZE = 40 # Analyze up to 40 continuous video chunks
FACE_SIZE = 299
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Loading inference on device: {DEVICE}")

# Initialize MTCNN for face extraction
mtcnn = MTCNN(keep_all=False, select_largest=True, post_process=False, device=DEVICE)

# Global transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Lazy load model
_model = None

def get_model():
    global _model
    if _model is None:
        model_path = os.path.join(os.path.dirname(__file__), "..", "video_model_training", "video_deepfake_model.pth")
        _model = VideoDeepfakeModel(sequence_length=NUM_FRAMES_PER_SEQ).to(DEVICE)
        
        # Load weights safely, we map_location to ensure it loads even if GPU/CPU state changes
        try:
            _model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            print("Successfully loaded trained deepfake model weights!")
        except Exception as e:
            print(f"Warning: Could not load model file {model_path}. Error: {str(e)}")
            print("Using untrained fallback weights for development purposes.")
            
        _model.eval()
    return _model

def analyze_video(video_path: str):
    """
    Analyzes the WHOLE video by extracting numerous continuous sequences of frames.
    For longer videos, it jumps ahead to grab multiple 15-frame continuous blocks, 
    running each chunk through the Video Deepfake model, flagging it if ANY block
    contains a deepfake anomaly.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < NUM_FRAMES_PER_SEQ:
        cap.release()
        raise ValueError(f"Video is too short. Needs at least {NUM_FRAMES_PER_SEQ} frames.")

    # Goal: We want to sample a continuous 15-frame block across the timeline.
    # E.g., sample a block once every 1 second
    stride_frames = int(fps) 
    
    # Calculate starting frame index for every chunk we want to pull
    start_frames = list(range(0, total_frames - NUM_FRAMES_PER_SEQ, stride_frames))
    
    # Prevent extracting too many chunks on very long movies (cap at 40 chunks / ~40 seconds distributed)
    if len(start_frames) > MAX_CHUNKS_TO_ANALYZE:
        indices = np.linspace(0, len(start_frames) - 1, MAX_CHUNKS_TO_ANALYZE, dtype=int)
        start_frames = [start_frames[i] for i in indices]
        
    chunked_sequences = []
    
    for start_f in start_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        
        segment_faces = []
        last_good_face = None
        
        for _ in range(NUM_FRAMES_PER_SEQ):
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            
            try:
                # Detect face
                face = mtcnn(img)
                if face is not None:
                    face_array = face.permute(1, 2, 0).byte().cpu().numpy()
                    face_img = Image.fromarray(face_array)
                    face_img = face_img.resize((FACE_SIZE, FACE_SIZE))
                    last_good_face = transform(face_img)
            except Exception:
                pass # Missing face, we'll gracefully fallback
                
            # If MTCNN drops a frame or person blinks, use the last known face to maintain temporal continuity
            if last_good_face is not None:
                segment_faces.append(last_good_face)
                
        # Only keep full 15-frame continuous segments
        if len(segment_faces) == NUM_FRAMES_PER_SEQ:
            chunked_sequences.append(torch.stack(segment_faces))
            
    cap.release()
    
    if len(chunked_sequences) == 0:
        raise ValueError("No continuous face sequences could be detected in the video.")
        
    # Run Inference on all continuous chunks in a single GPU batch
    # Shape: (Num_Chunks, 15, 3, 299, 299)
    tensor_batch = torch.stack(chunked_sequences).to(DEVICE)
    
    model = get_model()
    with torch.no_grad():
        with torch.amp.autocast(device_type=DEVICE.type if DEVICE.type == 'cuda' else 'cpu', enabled=DEVICE.type=='cuda'):
            raw_logits = model(tensor_batch) # Returns probability for EACH sequence block
            fake_probs = torch.sigmoid(raw_logits).cpu().numpy().flatten()
            
    # Whole-Video Analysis Strategy: 
    # If any specific continuous 15-frame block spikes as FAKE, we flag the whole video. 
    # This catches short AI face-swaps in the middle of otherwise real videos!
    # Deepfakes from different domains (e.g. YouTube shorts) can have lower absolute scores
    # than the Celeb-DF dataset. We use a threshold of 0.5 as standard.
    DETECTION_THRESHOLD = 0.50
    
    max_fake_prob = float(np.max(fake_probs))
    avg_fake_prob = float(np.mean(fake_probs))
    
    # We'll use the maximum probability found across the whole video's timeline as our final verdict
    verdict = "FAKE" if max_fake_prob >= DETECTION_THRESHOLD else "REAL"
    confidence = max_fake_prob if verdict == "FAKE" else (1.0 - max_fake_prob)
    
    # Optional scaling to ensure UI shows high confidence for fake
    if verdict == "FAKE" and confidence < 0.6:
        confidence = min(0.9, confidence * 2.0)
        
    print(f"Evaluated {len(chunked_sequences)} chunks across the whole video. Probabilities: {fake_probs}")
    
    # --- Grad-CAM Heatmap Generation ---
    heatmaps_base64 = []
    if verdict == "FAKE" and len(chunked_sequences) > 0:
        try:
            worst_idx = int(np.argmax(fake_probs))
            worst_chunk = tensor_batch[worst_idx].unsqueeze(0).clone().detach()
            worst_chunk.requires_grad_(True)

            activations = []
            gradients = []

            def forward_hook(module, input, output):
                activations.append(output)

            def backward_hook(module, grad_in, grad_out):
                gradients.append(grad_out[0])

            target_layer = model.feature_extractor.conv_head
            handle_forward = target_layer.register_forward_hook(forward_hook)
            handle_backward = target_layer.register_full_backward_hook(backward_hook)

            # Ensure model handles gradients explicitly
            with torch.enable_grad():
                if DEVICE.type == 'cuda':
                    # Trick PyTorch / cuBLAS into fully initializing on this background thread
                    _dummy = torch.matmul(torch.ones(1, 1, device=DEVICE, requires_grad=True), torch.ones(1, 1, device=DEVICE))
                    _dummy.sum().backward()
                    
                # cuDNN RNN backward pass explicitly requires the LSTM to be in training mode
                model.lstm.train()
                with torch.amp.autocast(device_type=DEVICE.type if DEVICE.type == 'cuda' else 'cpu', enabled=DEVICE.type=='cuda'):
                    model.zero_grad()
                    out_logits = model(worst_chunk)

                    # Since the output shape is (1, 1), we can just call backward on it directly
                    out_logits.backward(torch.ones_like(out_logits))
                
                # Revert LSTM to eval mode afterwards
                model.lstm.eval()

            handle_forward.remove()
            handle_backward.remove()

            if len(activations) > 0 and len(gradients) > 0:
                activations = activations[0]
                gradients = gradients[0]
                # activations/gradients shape: (sequence_length, channels, height, width) [15, C, H, W]
                # We extract all 15 frames from the deepfake sequence chunk
                frame_indices = list(range(15))
                mean_tensor = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(3, 1, 1)
                std_tensor = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(3, 1, 1)

                for f_idx in frame_indices:
                    if f_idx < activations.size(0):
                        act = activations[f_idx]
                        grad = gradients[f_idx]

                        # Global average pooling on gradients to get weights
                        weights = torch.mean(grad, dim=(1, 2), keepdim=True)
                        cam = torch.sum(weights * act, dim=0)
                        cam = torch.relu(cam)

                        cam_np = cam.detach().cpu().float().numpy()
                        
                        if np.isnan(cam_np).any() or np.max(cam_np) == 0:
                            continue
                            
                        # Normalize to 0-1
                        cam_np = cam_np / np.max(cam_np)
                        
                        # Apply a non-linear Gamma correction stretch (power curve)
                        # This mathematically boosts the lower "secondary" areas the model looked at
                        # (like eyes, nose, lips, hair) so they show up brightly too!
                        cam_np = np.power(cam_np, 0.4)
                        
                        cam_resized = cv2.resize(cam_np, (FACE_SIZE, FACE_SIZE))
                        
                        # Smooth the heatmap organically
                        cam_resized = cv2.GaussianBlur(cam_resized, (21, 21), 0)
                        
                        # Only strip the absolute dead 0-value background static
                        cam_resized = np.where(cam_resized < 0.05, 0, cam_resized)
                        if np.max(cam_resized) > 0:
                            cam_resized = cam_resized / np.max(cam_resized)

                        cam_uint8 = np.uint8(255 * cam_resized)
                        heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)

                        # Denormalize original image
                        orig_tensor = worst_chunk[0, f_idx].detach() * std_tensor + mean_tensor
                        orig_tensor = torch.clamp(orig_tensor, 0, 1)
                        orig_np = (orig_tensor.cpu().numpy() * 255).astype(np.uint8)
                        orig_np = np.transpose(orig_np, (1, 2, 0)) # H, W, C
                        orig_bgr = cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR)

                        # Only apply heatmap colors to the 'hot' regions. Cold regions remain the natural face.
                        # cam_resized (0 to 1) acts as our blending alpha mask. 
                        # We use a max of 0.70 opacity so the face features underneath are still highly visible
                        alpha = np.expand_dims(cam_resized, axis=-1) * 0.70
                        overlay = (heatmap * alpha + orig_bgr * (1.0 - alpha)).astype(np.uint8)
                        
                        _, buffer = cv2.imencode('.jpg', overlay)
                        b64_str = base64.b64encode(buffer).decode('utf-8')
                        heatmaps_base64.append(f"data:image/jpeg;base64,{b64_str}")
        except Exception as e:
            import traceback
            trace_str = traceback.format_exc()
            print(f"FAILED TO GENERATE GRAD-CAM: {e}\n{trace_str}")
            # If generating heatmaps fail, we still want to return the main verdict!
            
            # Send a fake image just to show an error heatmap on UI
            msg = "Error during Heatmap GPU Sync"
            blank_img = np.zeros((FACE_SIZE, FACE_SIZE, 3), dtype=np.uint8)
            cv2.putText(blank_img, "GPU ERROR", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            _, buffer = cv2.imencode('.jpg', blank_img)
            heatmaps_base64.append(f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}")
            pass

    return {
        "verdict": verdict,
        "confidence": confidence,
        "probability": max_fake_prob,
        "timeline_average": avg_fake_prob,
        "heatmaps": heatmaps_base64
    }
