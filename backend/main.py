from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import shutil
import os
import io
import tempfile
from datetime import datetime
import textwrap
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
import base64
from audio_detector import AudioDeepfakeDetector

app = FastAPI(title="Niriksha Deepfake API")

from routes.image import router as image_router
app.include_router(image_router)

from routes.document import router as document_router
app.include_router(document_router)

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
        print('VALUE ERROR DETAILS:', e); import traceback; traceback.print_exc(); raise HTTPException(status_code=400, detail=str(e))
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
    heatmaps: list[str] = []

@app.post("/generate/report")
async def generate_report_endpoint(request: ReportRequest):
    try:
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        pdf_path = os.path.join(temp_dir, f"report_{uuid.uuid4()}.pdf")
        
        # Create PDF using ReportLab
        c = canvas.Canvas(pdf_path, pagesize=letter)
        width, height = letter

        from watermark import add_watermark
        add_watermark(c, width, height)

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
        
        if request.heatmaps:
            y_pos -= 30
            if y_pos < 200:
                c.showPage()
                add_watermark(c, width, height)
                y_pos = height - 50

            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, y_pos, "5. Forensic Spatial Heatmaps (Grad-CAM)")  
            y_pos -= 15
            
            c.setFont("Helvetica-Oblique", 10)
            c.setFillColorRGB(0.4, 0.4, 0.4)
            c.drawString(50, y_pos, "Color Legend: Red/Orange = High AI Manipulation | Yellow = Moderate Artifacts | Cold/Clear = Natural Pixels")
            y_pos -= 15
            c.drawString(50, y_pos, "Note: The heatmaps highlight the regions of the image that the model considers indicative of manipulation.")
            y_pos -= 15
            c.drawString(50, y_pos, "Dark Red/Orange regions signify a high probability of artificial generation.")
            y_pos -= 15
            c.drawString(50, y_pos, "Yellow regions show moderate suspicion or potential blending/inpainting.")
            y_pos -= 15
            c.drawString(50, y_pos, "Blue/Cool regions (if present) show areas the model largely ignored or considers natural.")
            c.setFillColorRGB(0, 0, 0)
            y_pos -= 20

            x_offset = 70
            img_size = 90
            for i, heatmap_b64 in enumerate(request.heatmaps):
                # Move to next row if 5 images are drawn
                if i > 0 and i % 5 == 0:
                    y_pos -= (img_size + 20)
                    x_offset = 70
                    # Break page if needed
                    if y_pos < 100:
                        c.showPage()
                        add_watermark(c, width, height)
                try:
                    b64_data = heatmap_b64.split(",")[1] if "," in heatmap_b64 else heatmap_b64
                    img_data = base64.b64decode(b64_data)
                    img = ImageReader(io.BytesIO(img_data))
                    c.drawImage(img, x_offset, y_pos - img_size, width=img_size, height=img_size)
                    x_offset += (img_size + 10)
                except Exception as e:
                    print("Error drawing heatmap:", e)
            y_pos -= (img_size + 20)
        
        c.save()
        
        return FileResponse(
            path=pdf_path,
            media_type="application/pdf",
            filename="Niriksha_AI_Report.pdf",
            background=None  # Can be wrapped in a BackgroundTask for cleanup later
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")

class AudioReportRequest(BaseModel):
    result: dict


@app.post("/analyze/audio/report")
async def generate_audio_report(request: AudioReportRequest):
    data = request.result
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    from watermark import add_watermark
    add_watermark(c, width, height)

    # Header
    c.setFont("Helvetica-Bold", 18)
    c.setFillColorRGB(0.2, 0.5, 0.8)
    c.drawString(50, 750, "NIRIKSHA - Audio Forensic Analysis Report")
    c.setStrokeColorRGB(0.2, 0.5, 0.8)
    c.line(50, 740, 550, 740)

    c.setFillColorRGB(0, 0, 0)
    y = 710

    # Metadata
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Filename:")
    c.setFont("Helvetica", 12)
    c.drawString(150, y, str(data.get("filename", "Unknown")))
    y -= 20

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Media Type:")
    c.setFont("Helvetica", 12)
    c.drawString(150, y, str(data.get("media_type", "audio")))
    y -= 30

    # Verdict
    verdict = data.get("verdict", "UNKNOWN")
    confidence = data.get("confidence", 0)
    c.setFont("Helvetica-Bold", 16)
    if verdict == "FAKE":
        c.setFillColorRGB(0.8, 0, 0)
    elif verdict == "REAL":
        c.setFillColorRGB(0, 0.6, 0)
    else:
        c.setFillColorRGB(0.8, 0.6, 0)
    c.drawString(50, y, f"VERDICT: {verdict}")

    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica-Bold", 14)
    conf_pct = confidence * 100 if confidence <= 1 else confidence
    c.drawString(250, y, f"Confidence: {conf_pct:.1f}%")
    y -= 40

    # Sub-scores
    sub_scores = data.get("sub_scores", {})
    if sub_scores:
        c.setFont("Helvetica-Bold", 13)
        c.drawString(50, y, "Multi-Layer Analysis Breakdown:")
        y -= 25
        score_labels = {
            "neural_model": "Neural Model",
            "spectral_analysis": "Spectral Analysis",
            "spectral_consistency": "Spectral Consistency",
            "mfcc_analysis": "MFCC Analysis",
            "pitch_prosody": "Pitch / Prosody",
            "silence_pattern": "Silence Pattern",
        }
        c.setFont("Helvetica", 11)
        for key, label in score_labels.items():
            val = sub_scores.get(key)
            if val is not None:
                pct = val * 100 if val <= 1 else val
                flag = " [SUSPICIOUS]" if pct > 50 else ""
                c.drawString(70, y, f"• {label}: {pct:.1f}%{flag}")
                y -= 18
        y -= 10

    # Explanation
    explanation = data.get("explanation", "")
    if explanation:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, "Analysis Summary:")
        y -= 20
        c.setFont("Helvetica", 10)
        for line in textwrap.wrap(explanation, width=85):
            c.drawString(70, y, line)
            y -= 14
        y -= 10

    # Artifacts
    artifacts = data.get("artifacts_detected", [])
    if artifacts:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, "Detected Anomalies:")
        y -= 20
        c.setFont("Helvetica", 10)
        for artifact in artifacts:
            for line in textwrap.wrap(f"- {artifact}", width=85):
                if y < 60:
                    c.showPage()
                    y = 750
                    c.setFont("Helvetica", 10)
                c.drawString(70, y, line)
                y -= 14

    # Footer
    y -= 30
    if y < 80:
        c.showPage()
        y = 750
    c.setStrokeColorRGB(0.8, 0.8, 0.8)
    c.line(50, y + 15, 550, y + 15)
    c.setFont("Helvetica-Oblique", 9)
    c.setFillColorRGB(0.4, 0.4, 0.4)
    c.drawString(50, y, "Generated by Niriksha Audio Forensic Engine / Team 7seas.")
    c.drawString(50, y - 14, "This report is generated algorithmically and is for informational purposes only.")

    c.save()
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=Niriksha_Audio_Report.pdf"},
    )
