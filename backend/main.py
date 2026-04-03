from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import shutil
import os
import io
import tempfile
import textwrap
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
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


class AudioReportRequest(BaseModel):
    result: dict


@app.post("/analyze/audio/report")
async def generate_audio_report(request: AudioReportRequest):
    data = request.result
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

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

