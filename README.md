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
