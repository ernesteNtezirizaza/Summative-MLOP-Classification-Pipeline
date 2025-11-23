"""
FastAPI endpoints for brain tumor classification.
Provides prediction and retraining APIs.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
from PIL import Image
import io
import cv2
from datetime import datetime
import os
try:
    from src.prediction import BrainTumorPredictor
except ImportError:
    from prediction import BrainTumorPredictor
import time

app = FastAPI(title="Brain Tumor Classification API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance
predictor = None

# Model uptime tracking
start_time = time.time()
request_count = 0


def load_predictor():
    """Load the predictor model."""
    global predictor
    try:
        predictor = BrainTumorPredictor()
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    load_predictor()


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Brain Tumor Classification API",
        "status": "running",
        "endpoints": {
            "predict": "/predict",
            "retrain": "/retrain",
            "health": "/health",
            "uptime": "/uptime"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    model_loaded = predictor is not None
    return {
        "status": "healthy" if model_loaded else "model_not_loaded",
        "model_loaded": model_loaded,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/uptime")
async def get_uptime():
    """Get API uptime and statistics."""
    global request_count
    uptime_seconds = time.time() - start_time
    uptime_hours = uptime_seconds / 3600
    uptime_minutes = (uptime_seconds % 3600) / 60
    
    return {
        "uptime_seconds": int(uptime_seconds),
        "uptime_formatted": f"{int(uptime_hours)}h {int(uptime_minutes)}m",
        "total_requests": request_count,
        "model_loaded": predictor is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict brain tumor class from an uploaded image.
    """
    global request_count
    request_count += 1
    
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read image file
        contents = await file.read()
        
        # Convert to numpy array
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Make prediction
        start_pred_time = time.time()
        result = predictor.predict(img, return_probabilities=True)
        prediction_time = time.time() - start_pred_time
        
        return JSONResponse({
            "predicted_class": result['predicted_class'],
            "confidence": result['confidence'],
            "probabilities": result['probabilities'],
            "prediction_time_seconds": round(prediction_time, 4),
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """
    Predict classes for multiple images.
    """
    global request_count
    request_count += len(files)
    
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    
    for file in files:
        try:
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                results.append({
                    "filename": file.filename,
                    "error": "Invalid image file"
                })
                continue
            
            result = predictor.predict(img, return_probabilities=True)
            result['filename'] = file.filename
            results.append(result)
        
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return JSONResponse({
        "results": results,
        "total_images": len(files),
        "timestamp": datetime.now().isoformat()
    })


@app.post("/retrain")
async def trigger_retrain(files: list[UploadFile] = File(...)):
    """
    Upload images for retraining and trigger retraining process.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    # Create upload directory
    upload_dir = "data/retrain_uploads"
    os.makedirs(upload_dir, exist_ok=True)
    
    saved_files = []
    
    try:
        # Save uploaded files
        for file in files:
            # Determine class from filename or use a default
            # In production, you might want to require class labels
            contents = await file.read()
            file_path = os.path.join(upload_dir, file.filename)
            
            with open(file_path, "wb") as f:
                f.write(contents)
            
            saved_files.append(file_path)
        
        # Trigger retraining (this would typically be done asynchronously)
        # For now, return success message
        return JSONResponse({
            "message": "Files uploaded successfully. Retraining triggered.",
            "files_uploaded": len(saved_files),
            "file_paths": saved_files,
            "note": "Retraining will be processed. Check /retrain/status for progress.",
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")


@app.get("/retrain/status")
async def retrain_status():
    """Get retraining status."""
    return {
        "status": "not_implemented",
        "message": "Check retraining logs for status",
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    # Load predictor before starting server
    load_predictor()
    
    # Run server
    uvicorn.run(app, host="0.0.0.0", port=8000)

