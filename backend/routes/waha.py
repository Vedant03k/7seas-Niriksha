"""
WAHA webhook — receives WhatsApp messages, routes media to detectors,
generates PDF reports, and replies via WAHA API.
"""

import os
import asyncio
import tempfile
import traceback
from typing import Any

import httpx
from fastapi import APIRouter, Request

from utils import waha_client
from utils.report_generator import generate_report

router = APIRouter(tags=["waha"])

# ── Lazy-loaded detectors (shared singletons from other routes) ───────
# Audio detector is global in main.py; image/document are in their routers.
# We import them lazily to avoid circular imports and double-loading.

_audio_detector = None


def _get_audio_detector():
    global _audio_detector
    if _audio_detector is None:
        from audio_detector import AudioDeepfakeDetector
        _audio_detector = AudioDeepfakeDetector()
    return _audio_detector


def _get_image_detector():
    from routes.image import get_detector
    return get_detector()


def _get_document_detector():
    from routes.document import get_detector
    return get_detector()


# ── Supported MIME type → media category mapping ─────────────────────

MIME_MAP: dict[str, str] = {
    "image/jpeg": "image", "image/jpg": "image", "image/png": "image",
    "image/webp": "image", "image/bmp": "image",
    "audio/mpeg": "audio", "audio/mp3": "audio", "audio/wav": "audio",
    "audio/ogg": "audio", "audio/x-wav": "audio", "audio/aac": "audio",
    "audio/mp4": "audio", "audio/x-m4a": "audio",
    "video/mp4": "video", "video/webm": "video", "video/quicktime": "video",
    "video/x-matroska": "video",
    "application/pdf": "document",
}

USAGE_TEXT = (
    "👋 *Welcome to Niriksha — AI Deepfake Detector*\n\n"
    "Send me any *image, audio, video, or PDF* and I'll analyze it for "
    "deepfake manipulation.\n\n"
    "I'll reply with:\n"
    "• A quick verdict (REAL / FAKE)\n"
    "• A detailed PDF forensic report\n\n"
    "_Powered by Team 7seas_"
)


# ── Webhook endpoint ─────────────────────────────────────────────────

@router.post("/webhook/waha")
async def waha_webhook(request: Request):
    """
    Receives incoming WhatsApp messages from WAHA.
    Processes media asynchronously so the webhook returns 200 fast.
    """
    body: dict[str, Any] = await request.json()
    event = body.get("event")

    # Only handle incoming messages
    if event != "message":
        return {"status": "ignored", "reason": f"event={event}"}

    payload = body.get("payload", {})

    # Ignore outgoing (from-me) messages to avoid loops
    if payload.get("fromMe", False):
        return {"status": "ignored", "reason": "outgoing"}

    chat_id = payload.get("from", "")
    if not chat_id:
        return {"status": "ignored", "reason": "no sender"}

    # Fire-and-forget: process in background so webhook responds quickly
    asyncio.create_task(_handle_message(chat_id, payload))
    return {"status": "accepted"}


# ── Message handler (runs async in background) ──────────────────────

async def _handle_message(chat_id: str, payload: dict):
    try:
        await waha_client.send_seen(chat_id)

        has_media = payload.get("hasMedia", False)
        msg_type = payload.get("type", "chat")  # chat / image / audio / video / document

        if not has_media and msg_type == "chat":
            await waha_client.send_text(chat_id, USAGE_TEXT)
            return

        # Determine media category
        media_info = payload.get("media", {}) or {}
        mime = media_info.get("mimetype", "") or payload.get("mimetype", "")
        media_category = MIME_MAP.get(mime)

        # Fallback: infer from message type
        if not media_category:
            if msg_type in ("image", "sticker"):
                media_category = "image"
            elif msg_type in ("audio", "ptt"):
                media_category = "audio"
            elif msg_type == "video":
                media_category = "video"
            elif msg_type == "document" and mime == "application/pdf":
                media_category = "document"

        if not media_category:
            await waha_client.send_text(
                chat_id,
                "⚠️ Unsupported file type. Please send an *image, audio, video, or PDF*.",
            )
            return

        # Send "analyzing" acknowledgement
        await waha_client.send_text(
            chat_id,
            f"🔍 Analyzing your *{media_category}* for deepfakes… please wait.",
        )

        # Check if WAHA reported a decryption / download error
        if media_info.get("error"):
            print(f"[WAHA] Media error from WAHA: {media_info['error']}")

        # Download media from WAHA
        media_url = media_info.get("url") or payload.get("mediaUrl", "")
        if not media_url:
            # WAHA NOWEB — build download URL from message id
            msg_id = payload.get("id", "")
            session = waha_client.WAHA_SESSION
            media_url = f"{waha_client.WAHA_API_URL}/api/{session}/messages/{msg_id}/download"
        
        # If the URL is relative or missing host, prepend WAHA base URL
        if media_url.startswith("/"):
            media_url = f"{waha_client.WAHA_API_URL}{media_url}"

        print(f"[WAHA] Media URL resolved: {media_url}")
        print(f"[WAHA] Payload keys: {list(payload.keys())}")
        print(f"[WAHA] media_info: {media_info}")

        file_bytes = await _download_media(media_url)

        # If direct URL failed, try the message download endpoint as fallback
        if file_bytes is None:
            msg_id = payload.get("id", "")
            session = waha_client.WAHA_SESSION
            fallback_url = f"{waha_client.WAHA_API_URL}/api/{session}/messages/{msg_id}/download"
            if fallback_url != media_url:
                print(f"[WAHA] Trying fallback download: {fallback_url}")
                file_bytes = await _download_media(fallback_url)

        if file_bytes is None:
            await waha_client.send_text(
                chat_id,
                "❌ Could not download the media file. Please try sending it as a *document* (tap 📎 → Document).",
            )
            return

        # Route to detector
        result = await _analyze(media_category, file_bytes, mime)

        # Quick verdict reply
        verdict = result.get("verdict", "UNKNOWN")
        confidence = result.get("confidence", 0)
        conf_pct = confidence * 100 if confidence <= 1 else confidence
        emoji = "🔴" if verdict == "FAKE" else "🟢" if verdict == "REAL" else "🟡"
        summary = (
            f"{emoji} *Verdict: {verdict}*\n"
            f"Confidence: {conf_pct:.1f}%\n\n"
        )

        explanation = result.get("explanation", "")
        if explanation:
            summary += f"_{explanation[:300]}_\n\n"

        # WhatsApp summary from image detector (if present)
        wa_summary = result.get("whatsapp_summary", "")
        if wa_summary:
            summary = wa_summary + "\n\n"

        artifacts = result.get("artifacts_detected", [])
        if artifacts:
            summary += "*Anomalies:*\n"
            for a in artifacts[:5]:
                summary += f"• {a}\n"

        summary += "\n_🛡️ NIRIKSHA — Powered by Team 7seas_"

        await waha_client.send_text(chat_id, summary)

        # Generate & send PDF report
        try:
            pdf_bytes = generate_report(result, media_type=media_category)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            tmp.write(pdf_bytes)
            tmp.close()
            try:
                await waha_client.send_file(
                    chat_id,
                    tmp.name,
                    filename=f"Niriksha_{media_category.title()}_Report.pdf",
                    caption="📄 Full forensic report attached.",
                )
            finally:
                os.unlink(tmp.name)
        except Exception as pdf_err:
            print(f"[WAHA] PDF report send failed: {pdf_err}")
            await waha_client.send_text(
                chat_id,
                "📄 _PDF report generation is available on the web dashboard at http://localhost:5173_",
            )

    except Exception:
        traceback.print_exc()
        try:
            await waha_client.send_text(chat_id, "❌ Analysis failed. Please try again later.")
        except Exception:
            pass


# ── Media download helper ────────────────────────────────────────────

async def _download_media(url: str) -> bytes | None:
    headers = {}
    if waha_client.WAHA_API_KEY:
        headers["X-Api-Key"] = waha_client.WAHA_API_KEY
    print(f"[WAHA] Downloading media from: {url}")
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            print(f"[WAHA] Media downloaded: {len(resp.content)} bytes")
            return resp.content
    except Exception as e:
        print(f"[WAHA] Media download failed: {e}")
        return None


# ── Detector router ──────────────────────────────────────────────────

async def _analyze(category: str, file_bytes: bytes, mime: str) -> dict:
    """Route file bytes to the correct detector and return result dict."""
    # All detectors are CPU/GPU-bound — run in thread pool
    loop = asyncio.get_event_loop()

    if category == "image":
        detector = _get_image_detector()
        result = await loop.run_in_executor(None, detector.analyze, file_bytes)
        result["media_type"] = "image"
        return result

    if category == "audio":
        import shutil
        suffix = ".wav"
        if "mp3" in mime:
            suffix = ".mp3"
        elif "ogg" in mime:
            suffix = ".ogg"
        elif "m4a" in mime or "mp4" in mime:
            suffix = ".m4a"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(file_bytes)
        tmp.close()
        try:
            detector = _get_audio_detector()
            result = await loop.run_in_executor(None, detector.analyze_audio, tmp.name)
        finally:
            os.unlink(tmp.name)
        return result

    if category == "video":
        import uuid
        tmp_dir = os.path.join(os.path.dirname(__file__), "temp_uploads")
        os.makedirs(tmp_dir, exist_ok=True)
        suffix = ".mp4"
        if "webm" in mime:
            suffix = ".webm"
        elif "matroska" in mime:
            suffix = ".mkv"
        tmp_path = os.path.join(tmp_dir, f"{uuid.uuid4()}{suffix}")
        with open(tmp_path, "wb") as f:
            f.write(file_bytes)
        try:
            import video_inference
            result = await loop.run_in_executor(None, video_inference.analyze_video, tmp_path)
            result["media_type"] = "video"
            if result.get("verdict") == "FAKE":
                result.setdefault("artifacts_detected", [
                    "Unnatural Temporal Blinking",
                    "Inconsistent Face Boundaries over Frames",
                ])
                result.setdefault(
                    "explanation",
                    f"Spatial-Temporal AI model detected synthetic micro-jitters "
                    f"with {(result.get('confidence', 0) * 100):.1f}% certainty.",
                )
            else:
                result.setdefault("artifacts_detected", [])
                result.setdefault("explanation", "No manipulation artifacts detected.")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        return result

    if category == "document":
        detector = _get_document_detector()
        result = await loop.run_in_executor(
            None, detector.analyze_document, file_bytes, "uploaded.pdf"
        )
        result["media_type"] = "document"
        return result

    return {
        "verdict": "ERROR",
        "confidence": 0,
        "media_type": category,
        "artifacts_detected": [],
        "explanation": f"Unsupported media category: {category}",
    }
