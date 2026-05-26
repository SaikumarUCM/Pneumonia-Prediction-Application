from fastapi import FastAPI, File, UploadFile
import torch

from src.pneumonia_model import load_model
from src.core.utils import preprocess_image

app = FastAPI()

# Load model once at startup
model = load_model()

@app.get("/")
def home():
    return {"message": "Pneumonia Detection API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image = preprocess_image(file.file)

    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, dim=1).item()

    result = "PNEUMONIA" if prediction == 1 else "NORMAL"

    return {
        "prediction": result
    }
