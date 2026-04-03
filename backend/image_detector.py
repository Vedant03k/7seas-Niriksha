import io
import os
import base64
import hashlib
import datetime
import traceback
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ExifTags
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224

TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

AI_SOFTWARE_SIGNATURES = [
    "stable diffusion", "midjourney", "dall-e", "dalle",
    "generative", "runway", "adobe firefly", "deepfacelab",
    "faceswap", "reface", "avatarify", "gan", "synthesized",
    "artificial", "ai generated", "bing image creator"
]

class DeepfakeImageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.efficientnet_b4(
            weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1
        )
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        return self.backbone(x)

def load_model() -> DeepfakeImageModel:
    model = DeepfakeImageModel()
    weights_path = os.path.join(
        os.path.dirname(__file__), "models", "efficientnet_deepfake.pth"
    )
    if os.path.exists(weights_path):
        try:
            state_dict = torch.load(weights_path, map_location=DEVICE)
            model.load_state_dict(state_dict)
            print(f"[ImageDetector] Loaded finetuned weights from {weights_path}")
        except Exception as e:
            print(f"[ImageDetector] Could not load weights ({e}), using pretrained.")
    else:
        print("[ImageDetector] No finetuned weights found. Using ImageNet pretrained.")

    model.to(DEVICE)
    model.eval()
    return model

_MODEL: Optional[DeepfakeImageModel] = None

def get_model() -> DeepfakeImageModel:
    global _MODEL
    if _MODEL is None:
        _MODEL = load_model()
    return _MODEL

def analyze_frequency_domain(pil_img: Image.Image) -> dict:
    gray = np.array(pil_img.convert("L")).astype(np.float32)
    fft = np.fft.fft2(gray)
    fft_shifted = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shifted)
    log_magnitude = np.log1p(magnitude)

    h, w = log_magnitude.shape
    cy, cx = h // 2, w // 2

    mask = np.ones((h, w), dtype=bool)
    r_low = min(h, w) // 8
    y_grid, x_grid = np.ogrid[:h, :w]
    center_mask = (y_grid - cy) ** 2 + (x_grid - cx) ** 2 <= r_low ** 2
    mask[center_mask] = False

    total_energy = np.sum(log_magnitude)
    hf_energy = np.sum(log_magnitude[mask])
    spectral_peak_ratio = float(hf_energy / (total_energy + 1e-8))

    hf_region = log_magnitude.copy()
    hf_region[~mask] = 0
    hf_mean = np.mean(hf_region[mask])
    hf_std  = np.std(hf_region[mask])
    threshold = hf_mean + 4.5 * hf_std
    peak_pixels = np.sum(hf_region > threshold)
    has_gan_grid = bool(peak_pixels > 8)

    peak_ratio_score = min(float(peak_pixels) / 50.0, 1.0)
    hf_score = min(spectral_peak_ratio / 0.85, 1.0)
    fft_anomaly_score = float(0.6 * peak_ratio_score + 0.4 * hf_score)

    norm_mag = (log_magnitude - log_magnitude.min())
    norm_mag = norm_mag / (norm_mag.max() + 1e-8)
    heatmap_img = (norm_mag * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_INFERNO)
    heatmap_colored = cv2.resize(heatmap_colored, (256, 256))
    _, buffer = cv2.imencode(".png", heatmap_colored)
    fft_heatmap_b64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "fft_anomaly_score": round(fft_anomaly_score, 4),
        "spectral_peak_ratio": round(spectral_peak_ratio, 4),
        "has_gan_grid": has_gan_grid,
        "peak_pixel_count": int(peak_pixels),
        "fft_heatmap_b64": fft_heatmap_b64
    }

def generate_gradcam(
    model: DeepfakeImageModel,
    tensor: torch.Tensor,
    original_pil: Image.Image,
    target_class: int = 1
) -> str:
    try:
        target_layers = [model.backbone.features[-1]]
        cam = GradCAM(model=model, target_layers=target_layers)
        targets = [ClassifierOutputTarget(target_class)]

        grayscale_cam = cam(input_tensor=tensor.unsqueeze(0), targets=targets)
        grayscale_cam = grayscale_cam[0]

        orig_rgb = np.array(original_pil.convert("RGB").resize((IMG_SIZE, IMG_SIZE)))
        orig_normalized = orig_rgb.astype(np.float32) / 255.0

        overlay = show_cam_on_image(orig_normalized, grayscale_cam, use_rgb=True)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

        _, buffer = cv2.imencode(".png", overlay_bgr)
        return base64.b64encode(buffer).decode("utf-8")

    except Exception as e:
        print(f"[GradCAM] Error: {e}")
        return ""

def detect_face_regions(pil_img: Image.Image) -> list[dict]:
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        gray = np.array(pil_img.convert("L"))
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        if len(faces) == 0:
            return []
        return [{"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
                for (x, y, w, h) in faces]
    except Exception:
        return []

def analyze_metadata(pil_img: Image.Image, raw_bytes: bytes) -> dict:
    metadata_flags = []
    exif_data = {}
    ai_tool_detected = None
    suspicious = False

    try:
        exif_raw = pil_img._getexif()
        if exif_raw:
            for tag_id, value in exif_raw.items():
                tag = ExifTags.TAGS.get(tag_id, str(tag_id))
                try:
                    exif_data[tag] = str(value)[:200]
                except Exception:
                    pass

            software = exif_data.get("Software", "").lower()
            for sig in AI_SOFTWARE_SIGNATURES:
                if sig in software:
                    ai_tool_detected = exif_data.get("Software", "Unknown")
                    metadata_flags.append(f"AI software signature in EXIF: '{ai_tool_detected}'")
                    suspicious = True
                    break

            if "Make" not in exif_data and "Model" not in exif_data:
                metadata_flags.append("No camera make/model in EXIF (unusual for real photos)")

            if "DateTime" not in exif_data:
                metadata_flags.append("No capture timestamp in EXIF")

    except (AttributeError, Exception):
        metadata_flags.append("No EXIF data found (common in AI-generated images)")

    file_size_kb = len(raw_bytes) / 1024
    width, height = pil_img.size

    aspect = width / height if height > 0 else 1.0
    standard_ai_aspects = [1.0, 16/9, 4/3, 3/2, 9/16]
    is_exact_standard = any(abs(aspect - a) < 0.01 for a in standard_ai_aspects)
    if is_exact_standard and width in [512, 768, 1024, 1280, 1920]:
        metadata_flags.append(f"Resolution {width}x{height} matches common AI output dimensions")
        suspicious = True

    file_hash = hashlib.sha256(raw_bytes).hexdigest()

    return {
        "file_hash_sha256": file_hash,
        "file_size_kb": round(file_size_kb, 2),
        "image_dimensions": f"{width}x{height}",
        "aspect_ratio": round(aspect, 3),
        "exif_fields_found": len(exif_data),
        "exif_summary": {k: v for k, v in list(exif_data.items())[:8]},
        "ai_tool_detected": ai_tool_detected,
        "metadata_flags": metadata_flags,
        "metadata_suspicious": suspicious
    }

def classify_manipulation_type(
    model_confidence: float,
    fft_result: dict,
    face_count: int,
    metadata: dict
) -> dict:
    if model_confidence < 0.5:
        return {
            "type": "Likely Authentic",
            "description": "No significant manipulation signatures detected."
        }

    if fft_result["has_gan_grid"] and face_count > 0:
        return {
            "type": "GAN Face Synthesis",
            "description": (
                "Periodic GAN grid artifacts detected in frequency domain. "
                "The face region shows patterns consistent with StyleGAN or similar "
                "generative models that synthesize entire faces from noise."
            )
        }

    if metadata.get("ai_tool_detected"):
        return {
            "type": "AI-Generated Image",
            "description": (
                f"EXIF metadata contains AI software signature: "
                f"'{metadata['ai_tool_detected']}'. "
                "Image was likely generated or heavily processed by an AI tool."
            )
        }

    if fft_result["fft_anomaly_score"] > 0.6 and face_count > 0:
        return {
            "type": "FaceSwap / Face Reenactment",
            "description": (
                "Frequency domain anomalies detected around facial boundaries. "
                "Blending artifacts suggest the face region was replaced or "
                "re-animated using techniques like DeepFaceLab, FaceSwap, or "
                "First Order Motion Model."
            )
        }

    if model_confidence > 0.8:
        return {
            "type": "Neural Deepfake",
            "description": (
                "Neural network detected manipulation patterns. "
                "Specific technique unclear — could be face swap, "
                "attribute editing (aging, expression change), or "
                "inpainting-based manipulation."
            )
        }

    return {
        "type": "Possible Manipulation",
        "description": (
            "Moderate confidence of manipulation. "
            "Artifacts are subtle — could be light retouching or "
            "early-generation deepfake technique."
        )
    }

class ImageDeepfakeDetector:
    def __init__(self):
        print("[ImageDetector] Initializing...")
        self.model = get_model()
        print(f"[ImageDetector] Ready on device: {DEVICE}")

    def analyze(self, image_bytes: bytes) -> dict:
        timestamp = datetime.datetime.utcnow().isoformat() + "Z"

        try:
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            tensor = TRANSFORM(pil_img).to(DEVICE)

            with torch.no_grad():
                logits = self.model(tensor.unsqueeze(0))
                probs = F.softmax(logits, dim=1)[0]
                real_prob = float(probs[0])
                fake_prob = float(probs[1])

            fft_result = analyze_frequency_domain(pil_img)
            face_regions = detect_face_regions(pil_img)
            face_count = len(face_regions)
            metadata = analyze_metadata(pil_img, image_bytes)

            fft_contrib = fft_result["fft_anomaly_score"] * 0.30
            meta_contrib = 0.10 if metadata["metadata_suspicious"] else 0.0
            model_contrib = fake_prob * 0.60
            ensemble_score = min(model_contrib + fft_contrib + meta_contrib, 1.0)

            gradcam_b64 = ""
            if ensemble_score > 0.4 or face_count > 0:
                gradcam_b64 = generate_gradcam(self.model, tensor, pil_img)

            if ensemble_score >= 0.65:
                verdict = "FAKE"
            elif ensemble_score >= 0.45:
                verdict = "UNCERTAIN"
            else:
                verdict = "REAL"

            artifacts = []
            if fft_result["has_gan_grid"]:
                artifacts.append("GAN grid pattern (frequency domain)")
            if fft_result["fft_anomaly_score"] > 0.5:
                artifacts.append(f"Spectral anomaly score: {fft_result['fft_anomaly_score']:.2f}")
            if fake_prob > 0.7:
                artifacts.append(f"Neural network: {fake_prob * 100:.1f}% fake probability")
            if face_count == 0 and pil_img.size[0] > 100:
                artifacts.append("No face detected (may be full-scene synthesis)")
            if metadata["metadata_suspicious"]:
                artifacts.extend(metadata["metadata_flags"])
            if not artifacts:
                artifacts.append("No significant artifacts detected")

            manipulation = classify_manipulation_type(
                fake_prob, fft_result, face_count, metadata
            )

            explanation = _build_explanation(
                verdict, ensemble_score, fake_prob,
                fft_result, face_count, manipulation
            )

            return {
                "status": "success",
                "timestamp": timestamp,
                "verdict": verdict,
                "confidence": round(ensemble_score, 4),
                "confidence_percent": round(ensemble_score * 100, 1),

                "model_scores": {
                    "neural_network_fake_prob": round(fake_prob, 4),
                    "neural_network_real_prob": round(real_prob, 4),
                    "fft_anomaly_score": fft_result["fft_anomaly_score"],
                    "metadata_suspicious": metadata["metadata_suspicious"],
                    "ensemble_score": round(ensemble_score, 4)
                },

                "artifacts_detected": artifacts,
                "manipulation_type": manipulation,
                "explanation": explanation,

                "face_detection": {
                    "faces_found": face_count,
                    "face_regions": face_regions
                },

                "frequency_analysis": {
                    "fft_anomaly_score": fft_result["fft_anomaly_score"],
                    "spectral_peak_ratio": fft_result["spectral_peak_ratio"],
                    "has_gan_grid": fft_result["has_gan_grid"],
                    "peak_pixel_count": fft_result["peak_pixel_count"],
                    "fft_spectrum_image": fft_result["fft_heatmap_b64"]
                },

                "metadata_analysis": {
                    k: v for k, v in metadata.items()
                    if k != "exif_summary"
                },
                "exif_data": metadata.get("exif_summary", {}),

                "gradcam_heatmap": gradcam_b64,

                "legal": {
                    "file_hash_sha256": metadata["file_hash_sha256"],
                    "analysis_timestamp": timestamp,
                    "model_version": "DeepShield-ImageV1.0",
                    "tamper_proof_note": (
                        "SHA-256 hash generated on raw bytes at upload time. "
                        "This report can be verified against the original file."
                    )
                },

                "whatsapp_summary": _build_whatsapp_reply(
                    verdict, ensemble_score, manipulation, artifacts
                )
            }

        except Exception as e:
            return {
                "status": "error",
                "timestamp": timestamp,
                "error": str(e),
                "traceback": traceback.format_exc()
            }

def _build_explanation(
    verdict: str,
    ensemble_score: float,
    fake_prob: float,
    fft: dict,
    faces: int,
    manipulation: dict
) -> str:
    lines = []

    if verdict == "FAKE":
        lines.append(f"This image has been classified as FAKE with {ensemble_score*100:.1f}% confidence.")
    elif verdict == "UNCERTAIN":
        lines.append(f"This image shows mixed signals. Confidence: {ensemble_score*100:.1f}%.")
    else:
        lines.append(f"This image appears authentic with {(1-ensemble_score)*100:.1f}% real confidence.")

    lines.append(f"Manipulation type: {manipulation['type']}.")
    lines.append(manipulation["description"])

    if fft["has_gan_grid"]:
        lines.append(
            "The FFT spectrum shows periodic GAN upsampling artifacts — "
            "bright spots in the high-frequency zone indicate neural synthesis."
        )

    if faces == 0:
        lines.append("No human face was detected in this image.")
    elif faces == 1:
        lines.append("One face region detected and analyzed.")
    else:
        lines.append(f"{faces} face regions detected. Multi-face scenes can mask swap artifacts.")

    return " ".join(lines)

def _build_whatsapp_reply(
    verdict: str,
    confidence: float,
    manipulation: dict,
    artifacts: list
) -> str:
    icon = "❌" if verdict == "FAKE" else ("⚠️" if verdict == "UNCERTAIN" else "✅")
    art_summary = artifacts[0] if artifacts else "None"

    return (
        f"🔍 *DEEPSHIELD IMAGE REPORT*\n\n"
        f"*Verdict:* {icon} {verdict} ({confidence*100:.1f}% confidence)\n"
        f"*Type:* {manipulation['type']}\n"
        f"*Key Artifact:* {art_summary}\n\n"
        f"{'❌ Do NOT forward this media.' if verdict == 'FAKE' else '✅ Media appears authentic.'}\n"
        f"_Powered by DeepShield AI_"
    )
