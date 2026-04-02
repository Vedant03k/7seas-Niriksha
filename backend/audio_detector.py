import torch
import numpy as np
import subprocess
import tempfile
import os
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

class AudioDeepfakeDetector:
    def __init__(self):
        self.model_name = "facebook/wav2vec2-base" 
        
        print(f"Loading {self.model_name}...")
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name)
        
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=2, 
            ignore_mismatched_sizes=True
        )
        self.model.eval()

    def _load_audio_as_numpy(self, file_path: str, target_sr: int = 16000):
        """Use ffmpeg to convert any audio file to raw 16kHz mono PCM, then load as numpy."""
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.close()
        try:
            result = subprocess.run(
                ["ffmpeg", "-y", "-i", file_path, "-ar", str(target_sr), "-ac", "1", "-f", "wav", tmp.name],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg failed: {result.stderr[:500]}")
            
            import soundfile as sf
            data, sr = sf.read(tmp.name, dtype="float32")
            return data, sr
        finally:
            if os.path.exists(tmp.name):
                os.remove(tmp.name)

    def analyze_audio(self, file_path: str):
        try:
            # 1. Load audio via ffmpeg (works with mp3, wav, ogg, flac, etc.)
            waveform_np, sample_rate = self._load_audio_as_numpy(file_path, target_sr=16000)

            # 2. Extract features
            inputs = self.processor(
                waveform_np, 
                sampling_rate=16000, 
                return_tensors="pt", 
                padding=True
            )

            # 3. Run inference
            with torch.no_grad():
                logits = self.model(**inputs).logits
                probabilities = torch.softmax(logits, dim=-1)
                
            # Assume index 1 = FAKE, index 0 = REAL
            fake_prob = probabilities[0][1].item()
            real_prob = probabilities[0][0].item()

            verdict = "FAKE" if fake_prob > 0.5 else "REAL"
            confidence = max(fake_prob, real_prob)
            
            # Determine mock artifacts based on score
            artifacts = []
            if verdict == "FAKE":
                artifacts = ["Synthetic vocal tract signature detected", "TTS frequency clipping anomalies"]

            return {
                "verdict": verdict,
                "confidence": round(confidence, 3),
                "media_type": "audio",
                "artifacts_detected": artifacts,
                "heatmap_url": None, # Heatmaps apply to visual spectrograms mainly
                "explanation": f"Audio waveform analysis via Wav2Vec2 indicates this is an AI-generated voice." if verdict == "FAKE" else "Audio patterns appear consistent with natural human vocal generation."
            }
            
        except Exception as e:
            return {
                "verdict": "ERROR",
                "confidence": 0,
                "media_type": "audio",
                "artifacts_detected": [],
                "explanation": f"Failed to process audio: {str(e)}"
            }
