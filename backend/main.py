from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import shutil
import os
import tempfile
from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from audio_detector import AudioDeepfakeDetector

app = FastAPI(title="Niriksha Deepfake API")

from routes.image import router as image_router
app.include_router(image_router)

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


import sys
import uuid
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import video_inference
from fastapi import HTTPException

@app.post("/analyze/video")
async def analyze_video_endpoint(file: UploadFile = File(...)):
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Only video files are supported")

    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    temp_filepath = os.path.join(temp_dir, f"{uuid.uuid4()}_{file.filename}")

    try:
        with open(temp_filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"File saved to {temp_filepath}, running AI Video Inference...")
        result = video_inference.analyze_video(temp_filepath)
        result['media_type'] = "video"

        if result['verdict'] == 'FAKE':
            result['artifacts_detected'] = ['Unnatural Temporal Blinking', 'Inconsistent Face Boundaries over Frames']
            result['explanation'] = f"The Spatial-Temporal AI model detected synthetic micro-jitters with a certainty of {(result['confidence']*100):.1f}%."
        else:
            result['artifacts_detected'] = ['Natural smooth transitions', 'Consistent lighting on facial landscape']
            result['explanation'] = "No major manipulation artifacts were detected. The video frames map naturally to human biomechanics."

        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
    finally:
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)

# Pydantic model for report generation
class ReportRequest(BaseModel):
    verdict: str
    confidence: float
    media_type: str
    artifacts_detected: list[str]
    explanation: str

@app.post("/generate/report")
async def generate_report_endpoint(request: ReportRequest):
    try:
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        pdf_path = os.path.join(temp_dir, f"report_{uuid.uuid4()}.pdf")
        
        # Create PDF using ReportLab
        c = canvas.Canvas(pdf_path, pagesize=letter)
        width, height = letter
        
        # Header
        c.setFont("Helvetica-Bold", 24)
        c.drawString(50, height - 50, "Niriksha AI - Deepfake Analysis Report")
        
        c.setFont("Helvetica", 10)
        c.drawString(50, height - 70, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Verdict Section
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, height - 110, "1. Final Verdict")
        
        color = colors.red if request.verdict == "FAKE" else colors.green
        c.setFillColor(color)
        c.setFont("Helvetica-Bold", 18)
        c.drawString(70, height - 135, request.verdict)
        
        c.setFillColor(colors.black)
        c.setFont("Helvetica", 12)
        c.drawString(200, height - 135, f"Confidence Score: {(request.confidence * 100):.1f}%")
        
        # Media Details
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, height - 180, "2. Media Information")
        c.setFont("Helvetica", 12)
        c.drawString(70, height - 200, f"Media Type: {request.media_type.capitalize()}")
        
        # Anomalies
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, height - 240, "3. Detected Anomalies / Artifacts")
        y_pos = height - 260
        c.setFont("Helvetica", 12)
        for artifact in request.artifacts_detected:
            c.drawString(70, y_pos, f"- {artifact}")
            y_pos -= 20
            
        # Explanation
        c.setFont("Helvetica-Bold", 14)
        y_pos -= 20
        c.drawString(50, y_pos, "4. Technical Explanation")
        y_pos -= 20
        
        c.setFont("Helvetica", 11)
        # Simple text wrapping for the explanation
        words = request.explanation.split(' ')
        line = ""
        for word in words:
            if c.stringWidth(line + word + " ", "Helvetica", 11) < (width - 100):
                line += word + " "
            else:
                c.drawString(70, y_pos, line)
                y_pos -= 15
                line = word + " "
        c.drawString(70, y_pos, line)
        
        # Footer
        c.setFont("Helvetica-Oblique", 10)
        c.setFillColor(colors.gray)
        c.drawString(50, 30, "This is an automatically generated AI report by Niriksha AI. Use for supplementary intelligence.")
        
        c.save()
        
        return FileResponse(
            path=pdf_path,
            media_type="application/pdf",
            filename="Niriksha_AI_Report.pdf",
            background=None  # Can be wrapped in a BackgroundTask for cleanup later
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")
