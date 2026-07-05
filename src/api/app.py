import os
from typing import List

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.core.utils import preprocess_image_to_numpy

app = FastAPI(title="Pneumonia Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ONNX_MODEL_PATH = os.getenv("ONNX_MODEL_PATH", "modelWeights/pneumonia_model.onnx")

session = ort.InferenceSession(
    ONNX_MODEL_PATH,
    providers=["CPUExecutionProvider"],
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name


def softmax(x: np.ndarray) -> np.ndarray:
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=0)


def _format_prediction(probs: np.ndarray) -> dict:
    prediction = int(np.argmax(probs))
    confidence = float(probs[prediction])
    result = "PNEUMONIA" if prediction == 1 else "NORMAL"
    return {
        "prediction": result,
        "confidence": confidence,
        "prediction_index": prediction,
    }


def _run_inference(batch_np: np.ndarray) -> List[dict]:
    output = session.run([output_name], {input_name: batch_np})[0]
    return [_format_prediction(softmax(output[i])) for i in range(output.shape[0])]


@app.get("/")
def home():
    return {"message": "Pneumonia Detection API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_np = preprocess_image_to_numpy(file.file).astype(np.float32)
    return _run_inference(image_np)[0]


@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required.")

    batch_tensors = []
    filenames = []

    for upload in files:
        batch_tensors.append(preprocess_image_to_numpy(upload.file))
        filenames.append(upload.filename or "unknown")

    batch_np = np.concatenate(batch_tensors, axis=0).astype(np.float32)
    predictions = _run_inference(batch_np)

    return {
        "count": len(predictions),
        "results": [
            {"filename": filename, **prediction}
            for filename, prediction in zip(filenames, predictions)
        ],
    }
