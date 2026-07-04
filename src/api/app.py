from fastapi import FastAPI, File, UploadFile
import torch
import onnxruntime as ort
import numpy as np

from src.pneumonia_model import load_model
from src.core.utils import preprocess_image

app = FastAPI()

ONNX_MODEL_PATH = "modelWeights/pneumonia_model.onnx"


# Load Onnx model once at startup
session= ort.InferenceSession(
    ONNX_MODEL_PATH,
    providers=["CPUExecutionProvider"],
)

# Confirm input/output names match your exported model
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name


def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)




@app.get("/")
def home():
    return {"message": "Pneumonia Detection API is running"}




@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image = preprocess_image(file.file)          # Torch tensor [1,3,224,224]
    image_np=image.numpy().astype(np.float32)    # ONNX expects numpy float32

    output = session.run([output_name], {input_name: image_np})[0]
    probs= softmax(output[0])


    prediction = int(np.argmax(probs))
    confidence = float(probs[prediction])
    result = "PNEUMONIA" if prediction == 1 else "NORMAL"
    return {
        "prediction": result,
        "confidence": confidence,
        "prediction_index": prediction
    }


