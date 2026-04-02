from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import tempfile
from audio_detector import AudioDeepfakeDetector

app = FastAPI(title="Niriksha Deepfake API")

# CORS - allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models only once when API starts up
print("Loading Deepfake Detection Models...")
audio_detector = AudioDeepfakeDetector()

@app.get("/")
def read_root():
    return {"status": "active", "message": "Deepfake Detection API is running"}

@app.post("/analyze/audio")
async def analyze_audio_endpoint(file: UploadFile = File(...)):
    if not file.filename:
        return {"verdict": "ERROR", "confidence": 0, "media_type": "audio",
                "artifacts_detected": [], "explanation": "No file uploaded"}
    
    # Save to OS temp directory (outside project) to avoid triggering uvicorn reload
    suffix = os.path.splitext(file.filename)[1] or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_path = tmp.name
    try:
        shutil.copyfileobj(file.file, tmp)
        tmp.close()
        result = audio_detector.analyze_audio(temp_path)
    except Exception as e:
        result = {
            "verdict": "ERROR", "confidence": 0, "media_type": "audio",
            "artifacts_detected": [], "explanation": f"Server error: {str(e)}"
        }
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
    return result

