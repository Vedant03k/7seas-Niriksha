# Niriksha-deepfake

## Team: 7seas

- **Leader:** Vedant Kale
- **Members:** Amrita Kumari, Pruthviraj Rajput, Omkar Madakar

## System Architecture & Setup

This repository contains the full stack implementation of a deepfake detection system.

### Current Setup & Installed Dependencies

The project is structured into two main directories: `backend` and `frontend`. Initial setup and dependency installations have been completed.

#### Backend (Python/FastAPI & PyTorch)

- **Framework & Asynchronous Queue**: `fastapi`, `uvicorn`, `celery`, `redis`
- **Machine Learning & Vision**: `torch`, `torchvision`, `torchaudio`, `transformers` (Hugging Face), `opencv-python`, `onnxruntime`, `dlib`, `numpy`
- **Data Storage & Infra**: `psycopg2-binary` (PostgreSQL), `boto3` (MinIO/S3), `mlflow` (Experiment tracking), `prometheus_client` (Metrics)

#### Frontend (React + Vite, TypeScript)

- **Framework & Styling**: React with Vite (`react-ts` template), `tailwindcss`, `postcss`, `autoprefixer`
- **Charting & Functionality**: `chart.js`, `react-chartjs-2` (Heatmaps/visualizations), `axios` (API polling)

---

## WhatsApp Integration (WAHA)

Niriksha integrates with [WAHA (WhatsApp HTTP API)](https://waha.devlike.pro/) to provide deepfake detection directly inside WhatsApp. Users can send any image, audio, video, or PDF to the connected WhatsApp number and receive an instant AI-powered verdict along with a detailed PDF forensic report — no app or website visit needed.

### How It Works

1. **WAHA** is a self-hosted WhatsApp gateway that exposes a REST API and delivers incoming messages to a webhook.
2. The FastAPI backend registers a webhook endpoint at **`POST /webhook/waha`** (see `backend/routes/waha.py`).
3. When a user sends media on WhatsApp:
   - The webhook receives the message payload from WAHA.
   - The media file (image / audio / video / PDF) is downloaded via the WAHA API.
   - The file is routed to the appropriate detector (image, audio, video, or document).
   - A quick text verdict (*REAL* / *FAKE* with confidence %) is sent back to the user.
   - A full PDF forensic report is generated and delivered as a WhatsApp document.
4. If a plain text message is received, the bot replies with usage instructions.

### Key Files

| File | Purpose |
|------|---------|
| `backend/routes/waha.py` | Webhook endpoint, media routing, verdict & report delivery |
| `backend/utils/waha_client.py` | WAHA API client — sends text messages, files, and read receipts |
| `backend/utils/report_generator.py` | Generates the PDF forensic report attached to replies |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WAHA_API_URL` | `http://localhost:3000` | Base URL of the WAHA instance |
| `WAHA_SESSION` | `default` | WAHA session name |
| `WAHA_API_KEY` | *(set in code)* | API key for authenticating with WAHA |

### Running WAHA Locally

```bash
# Pull and run the WAHA Docker container
docker run -d --name waha -p 3000:3000 devlikeapro/waha

# Start the Niriksha backend (registers the webhook automatically)
cd backend
uvicorn main:app --reload --port 8000
```

Configure WAHA to send webhooks to `http://<backend-host>:8000/webhook/waha`. Scan the QR code shown in the WAHA dashboard to link your WhatsApp account.

### Supported Media Types

| Category | MIME Types |
|----------|-----------|
| **Image** | `image/jpeg`, `image/png`, `image/webp`, `image/bmp` |
| **Audio** | `audio/mpeg`, `audio/wav`, `audio/ogg`, `audio/aac`, `audio/mp4`, `audio/x-m4a` |
| **Video** | `video/mp4`, `video/webm`, `video/quicktime`, `video/x-matroska` |
| **Document** | `application/pdf` |
