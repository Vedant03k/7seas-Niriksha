from fastapi import FastAPI

app = FastAPI(title="Niriksha Deepfake API")

@app.get(/")
def read_root():
    return {"status": "active", "message": "Deepfake Detection API is running"}

