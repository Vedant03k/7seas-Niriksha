"""
WAHA API client — sends text messages and PDF files back via WhatsApp.

Expects WAHA running at WAHA_API_URL (default http://localhost:3000).
"""

import os
import httpx

WAHA_API_URL = os.getenv("WAHA_API_URL", "http://localhost:3000")
WAHA_SESSION = os.getenv("WAHA_SESSION", "default")
WAHA_API_KEY = os.getenv("WAHA_API_KEY", "2c0ce850c25a48058ccfceb3b37ee423")

_HEADERS = {"Content-Type": "application/json"}
if WAHA_API_KEY:
    _HEADERS["X-Api-Key"] = WAHA_API_KEY


async def send_text(chat_id: str, text: str) -> dict:
    """Send a plain-text WhatsApp message."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{WAHA_API_URL}/api/sendText",
            headers=_HEADERS,
            json={
                "session": WAHA_SESSION,
                "chatId": chat_id,
                "text": text,
            },
        )
        resp.raise_for_status()
        return resp.json()


async def send_file(chat_id: str, file_path: str, filename: str, caption: str = "") -> dict:
    """Send a file (PDF report) as a WhatsApp document."""
    with open(file_path, "rb") as f:
        file_bytes = f.read()

    import base64
    file_b64 = base64.b64encode(file_bytes).decode()

    payload = {
        "session": WAHA_SESSION,
        "chatId": chat_id,
        "file": {
            "mimetype": "application/pdf",
            "filename": filename,
            "data": f"data:application/pdf;base64,{file_b64}",
        },
        "caption": caption,
    }

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{WAHA_API_URL}/api/sendFile",
            headers=_HEADERS,
            json=payload,
        )
        print(f"[WAHA] sendFile response: {resp.status_code} {resp.text[:200]}")
        resp.raise_for_status()
        return resp.json()


async def send_seen(chat_id: str) -> None:
    """Mark chat as seen/read."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(
                f"{WAHA_API_URL}/api/sendSeen",
                headers=_HEADERS,
                json={"session": WAHA_SESSION, "chatId": chat_id},
            )
    except Exception:
        pass
