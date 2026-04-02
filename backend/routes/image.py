import io
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

from image_detector import ImageDeepfakeDetector

router = APIRouter()

_detector: ImageDeepfakeDetector | None = None

def get_detector() -> ImageDeepfakeDetector:
    global _detector
    if _detector is None:
        _detector = ImageDeepfakeDetector()
    return _detector

ALLOWED_TYPES = {
    "image/jpeg", "image/jpg", "image/png",
    "image/webp", "image/bmp", "image/tiff"
}
MAX_FILE_SIZE_MB = 20

@router.get("/analyze/image/health")
async def image_health():
    try:
        detector = get_detector()
        return {
            "status": "ok",
            "model": "EfficientNet-B4 + FFT + GradCAM",
            "device": str(detector.model.backbone.parameters().__next__().device),
            "message": "Image detector ready"
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": str(e)}
        )

@router.post("/analyze/image")
async def analyze_image(file: UploadFile = File(...)):
    content_type = file.content_type or ""
    if content_type not in ALLOWED_TYPES:
        filename = (file.filename or "").lower()
        if not any(filename.endswith(ext) for ext in
                   [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"]):
            raise HTTPException(
                status_code=415,
                detail=(
                    f"Unsupported file type: '{content_type}'. "
                    f"Accepted types: JPEG, PNG, WEBP, BMP, TIFF"
                )
            )

    image_bytes = await file.read()
    size_mb = len(image_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {size_mb:.1f} MB. Max: {MAX_FILE_SIZE_MB} MB"
        )

    if len(image_bytes) < 100:
        raise HTTPException(
            status_code=400,
            detail="File appears to be empty or corrupt."
        )

    detector = get_detector()
    result = detector.analyze(image_bytes)

    if result.get("status") == "error":
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Detection pipeline failed",
                "error": result.get("error"),
                "traceback": result.get("traceback")
            }
        )

    result["filename"] = file.filename or "unknown"
    result["file_size_mb"] = round(size_mb, 3)

    return JSONResponse(content=result)

class ReportRequest(BaseModel):
    result: dict

@router.post("/analyze/image/report")
async def generate_pdf_report(request: ReportRequest):
    data = request.result
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    
    # Header
    c.setFont("Helvetica-Bold", 18)
    c.setStrokeColorRGB(0.2, 0.5, 0.8)
    c.drawString(50, 750, "DEEPSHIELD - Certified Forensic Analysis Report")
    c.line(50, 740, 550, 740)
    
    # Metadata
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, 710, "Target Media:")
    c.setFont("Helvetica", 12)
    c.drawString(150, 710, str(data.get('filename', 'Unknown')))
    
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, 690, "Timestamp UTC:")
    c.setFont("Helvetica", 12)
    c.drawString(150, 690, str(data.get('legal', {}).get('analysis_timestamp', '')))
    
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, 670, "SHA-256 Hash:")
    c.setFont("Courier", 10)
    c.drawString(150, 670, str(data.get('legal', {}).get('file_hash_sha256', '')))
    
    # Verdict
    verdict = data.get('verdict', 'UNKNOWN')
    conf = data.get('confidence_percent', 0)
    c.setFont("Helvetica-Bold", 16)
    if verdict == "FAKE":
        c.setFillColorRGB(0.8, 0, 0)
    elif verdict == "REAL":
        c.setFillColorRGB(0, 0.6, 0)
    else:
        c.setFillColorRGB(0.8, 0.6, 0)
    c.drawString(50, 630, f"VERDICT: {verdict}")
    
    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(250, 630, f"AI Confidence Score: {conf}%")
    
    # Manipulation Type
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, 590, "Primary Manipulation Classification:")
    c.setFont("Helvetica", 11)
    type_str = data.get('manipulation_type', {}).get('type', '')
    desc_str = data.get('manipulation_type', {}).get('description', '')
    c.drawString(70, 570, type_str)
    
    # Wrap text for description
    import textwrap
    y = 550
    for line in textwrap.wrap(desc_str, width=80):
        c.drawString(70, y, line)
        y -= 15
        
    # Anomalies
    y -= 15
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Digital Artifacts & Anomalies Detected:")
    y -= 20
    c.setFont("Helvetica", 11)
    for artifact in data.get('artifacts_detected', []):
        c.drawString(70, y, f"- {artifact}")
        y -= 15
        
    # Legal Footer
    y -= 40
    c.setStrokeColorRGB(0.8, 0.8, 0.8)
    c.line(50, y+15, 550, y+15)
    c.setFont("Helvetica-Oblique", 10)
    c.setFillColorRGB(0.4, 0.4, 0.4)
    c.drawString(50, y, "TAMPER-PROOF RECORD: This document is tied to the SHA-256 fingerprint shown above.")
    c.drawString(50, y-15, f"DeepShield Engine Version: {data.get('legal', {}).get('model_version', '1.0')}")
    c.drawString(50, y-30, "Generated algorithmically by Niriksha / Team 7seas.")
    
    c.save()
    buffer.seek(0)
    
    file_id = str(data.get('legal', {}).get('file_hash_sha256', 'report'))[:8]
    return StreamingResponse(
        buffer, 
        media_type="application/pdf", 
        headers={"Content-Disposition": f"attachment; filename=DeepShield_Report_{file_id}.pdf"}
    )
