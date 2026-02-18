from fastapi import FastAPI, UploadFile, File
import os
import shutil
from tracking import track_video
from analysis import analyze_dataframe

app = FastAPI()


@app.get("/")
def health_check():
    return {"status": "Backend running."}


@app.post("/analyze-video")
async def analyze_video(file: UploadFile = File(...)):
    """
    Accepts video file from frontend.
    Runs tracking.
    Returns analysis JSON.
    """

    temp_path = f"temp_{file.filename}"

    # Save uploaded file temporarily
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run tracking
    df = track_video(temp_path)

    # Analyze
    result = analyze_dataframe(df)

    # Remove temp file
    os.remove(temp_path)

    return result
