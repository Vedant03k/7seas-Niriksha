import io
import textwrap
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from document_detector import DocumentForensicDetector

router = APIRouter()

_detector: DocumentForensicDetector | None = None

def get_detector() -> DocumentForensicDetector:
    global _detector
    if _detector is None:
        _detector = DocumentForensicDetector()
    return _detector

ALLOWED_TYPES = {
    "application/pdf",
}
MAX_FILE_SIZE_MB = 50


@router.get("/analyze/document/health")
async def document_health():
    try:
        detector = get_detector()
        return {
            "status": "ok",
            "model": "Multi-layer PDF Forensics (Metadata + Structure + ELA + Text + Signature)",
            "message": "Document detector ready",
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": str(e)},
        )


@router.post("/analyze/document")
async def analyze_document(file: UploadFile = File(...)):
    content_type = file.content_type or ""
    filename = file.filename or "document.pdf"

    if content_type not in ALLOWED_TYPES:
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type: '{content_type}'. Only PDF files are accepted.",
            )

    file_bytes = await file.read()
    size_mb = len(file_bytes) / (1024 * 1024)

    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {size_mb:.1f} MB. Max: {MAX_FILE_SIZE_MB} MB",
        )

    if len(file_bytes) < 100:
        raise HTTPException(
            status_code=400,
            detail="File appears to be empty or corrupt.",
        )

    # Validate it starts with PDF magic bytes
    if not file_bytes[:5].startswith(b"%PDF-"):
        raise HTTPException(
            status_code=400,
            detail="File does not appear to be a valid PDF.",
        )

    detector = get_detector()
    result = detector.analyze_document(file_bytes, filename)

    if result.get("verdict") == "ERROR":
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Document analysis failed",
                "error": result.get("explanation"),
            },
        )

    result["file_size_mb"] = round(size_mb, 3)
    return JSONResponse(content=result)


class DocumentReportRequest(BaseModel):
    result: dict


@router.post("/analyze/document/report")
async def generate_document_report(request: DocumentReportRequest):
    data = request.result
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from watermark import add_watermark
    add_watermark(c, width, height)

    # Header
    c.setFont("Helvetica-Bold", 18)
    c.setFillColorRGB(0.2, 0.5, 0.8)
    c.drawString(50, 750, "NIRIKSHA - Document Forensic Analysis Report")
    c.setStrokeColorRGB(0.2, 0.5, 0.8)
    c.line(50, 740, 550, 740)

    c.setFillColorRGB(0, 0, 0)
    y = 710

    # Metadata
    meta = data.get("metadata_info", {})
    fields = [
        ("Filename", data.get("filename", "Unknown")),
        ("Producer", meta.get("producer", "N/A")),
        ("Creator", meta.get("creator", "N/A")),
        ("Author", meta.get("author", "N/A")),
        ("SHA-256", meta.get("file_hash_sha256", "N/A")),
        ("File Size", f"{meta.get('file_size_kb', 'N/A')} KB"),
    ]
    for label, value in fields:
        c.setFont("Helvetica-Bold", 11)
        c.drawString(50, y, f"{label}:")
        c.setFont("Helvetica", 11)
        val_str = str(value)
        c.drawString(150, y, val_str[:70])
        y -= 18
    y -= 10

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
            "metadata_forensics": "Metadata Forensics",
            "structural_analysis": "Structural Analysis",
            "visual_ela": "Visual / ELA",
            "text_consistency": "Text Consistency",
            "digital_signature": "Digital Signature",
        }
        c.setFont("Helvetica", 11)
        for key, label in score_labels.items():
            val = sub_scores.get(key)
            if val is not None:
                pct = val * 100 if val <= 1 else val
                flag = " [SUSPICIOUS]" if pct > 30 else ""
                c.drawString(70, y, f"\u2022 {label}: {pct:.1f}%{flag}")
                y -= 18
        y -= 10

    # Structure info
    struct = data.get("structure_info", {})
    if struct:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, "Document Structure:")
        y -= 20
        c.setFont("Helvetica", 11)
        struct_items = [
            f"Pages: {struct.get('page_count', 'N/A')}",
            f"Fonts: {struct.get('total_fonts', 'N/A')}",
            f"Images: {struct.get('total_images', 'N/A')}",
            f"Annotations: {struct.get('total_annotations', 'N/A')}",
            f"Image-only: {'Yes' if struct.get('is_image_only') else 'No'}",
        ]
        for item in struct_items:
            c.drawString(70, y, f"\u2022 {item}")
            y -= 16
        y -= 10

    # Signature info
    sig = data.get("signature_info", {})
    if sig:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, "Digital Signature:")
        y -= 20
        c.setFont("Helvetica", 11)
        has_sig = sig.get("has_digital_signature", False)
        c.drawString(70, y, f"\u2022 Signature present: {'Yes' if has_sig else 'No'}")
        y -= 16
        if has_sig:
            c.drawString(70, y, f"\u2022 Signature count: {sig.get('signature_count', 0)}")
            y -= 16
            c.drawString(70, y, f"\u2022 Certificate chain: {'Yes' if sig.get('has_certificate_chain') else 'No'}")
            y -= 16
        y -= 10

    # Explanation
    explanation = data.get("explanation", "")
    if explanation:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, "Analysis Summary:")
        y -= 20
        c.setFont("Helvetica", 10)
        for line in textwrap.wrap(explanation, width=85):
            if y < 60:
                c.showPage()
                y = 750
                c.setFont("Helvetica", 10)
            c.drawString(70, y, line)
            y -= 14
        y -= 10

    # Artifacts
    artifacts = data.get("artifacts_detected", [])
    if artifacts:
        if y < 100:
            c.showPage()
            y = 750
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
    c.drawString(50, y, "Generated by Niriksha Document Forensic Engine / Team 7seas.")
    c.drawString(50, y - 14, "This report is generated algorithmically and is for informational purposes only.")

    c.save()
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=Niriksha_Document_Report.pdf"},
    )
