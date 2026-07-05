# Pneumonia Prediction App

An AI-powered application designed to detect pneumonia from chest X-ray images using deep learning. This project implements a complete pipeline from PyTorch model training to **ONNX Runtime inference** in a deployable web application.

## 🚀 Features
- **Deep Learning Model**: Uses a fine-tuned ResNet18 architecture for high-accuracy binary classification (Normal vs. Pneumonia).
- **ONNX Inference**: Production API serves predictions via **ONNX Runtime** for faster, portable inference.
- **FastAPI Backend**: A robust REST API that handles image uploads and returns predictions with confidence scores.
- **Streamlit Frontend**: A user-friendly web interface for uploading X-ray images and viewing real-time predictions.
- **Pre-processing Pipeline**: Integrated image normalization and resizing to ensure consistent model input.
- **Docker Deployment**: One-command setup with Docker Compose (API + Streamlit).
- **Cloud Ready**: Deploy to [Render](https://render.com) or [Hugging Face Spaces](https://huggingface.co/spaces) for a live demo link.

## 📸 Screenshots

![App Overview](utils/screenshot-1.png)

![Detailed View](utils/screenshot-2.png)

## 🛠️ Tech Stack
- **Language**: Python
- **Deep Learning Framework**: PyTorch, Torchvision (training & export)
- **Inference Runtime**: ONNX Runtime
- **Model Format**: ONNX (`.onnx` + external weights `.onnx.data`)
- **Backend Framework**: FastAPI
- **Frontend Framework**: Streamlit
- **Image Processing**: PIL (Pillow)
- **Metrics**: Scikit-learn (Accuracy Score)
- **Deployment**: Docker, Docker Compose, Render, Hugging Face Spaces

## 📂 Project Structure
```text
.
├── Model Notebooks/       # Training scripts and dataset
│   ├── data/              # Chest X-ray dataset (train, test, val)
│   └── train_model.py     # Model training and evaluation script
├── modelWeights/          # Saved model artifacts
│   ├── pneumonia_model.pth        # PyTorch weights (training / re-export)
│   ├── pneumonia_model.onnx       # ONNX model graph
│   └── pneumonia_model.onnx.data  # ONNX external weights
├── src/                   # Source code
│   ├── api/               # FastAPI implementation
│   │   └── app.py         # API endpoints (ONNX Runtime inference)
│   ├── core/              # Core utility functions
│   │   └── utils.py       # Image preprocessing logic
│   ├── inference_onnx.py  # One-time PyTorch → ONNX export script
│   └── pneumonia_model.py # Model architecture and PyTorch loading
├── streamlit_app.py       # Streamlit frontend application
├── Dockerfile.api         # Docker image for FastAPI + ONNX
├── Dockerfile.streamlit   # Docker image for Streamlit UI
├── docker-compose.yml     # Run API + Streamlit together
├── render.yaml            # Render.com blueprint (live demo)
├── requirements.txt       # Full local development dependencies
├── requirements-api.txt   # Slim API-only dependencies (Docker)
├── requirements-ui.txt    # Streamlit-only dependencies (Docker)
└── pyproject.toml         # Project configuration
```

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd "Pneominia Prediction"
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Export the Model to ONNX (first time or after retraining)
If `modelWeights/pneumonia_model.onnx` is not present, export from the trained PyTorch weights:
```bash
python -m src.inference_onnx
```
This creates `pneumonia_model.onnx` and `pneumonia_model.onnx.data` in `modelWeights/`. Keep both files together.

### 4. Run the Backend API
Start the FastAPI server:
```bash
uvicorn src.api.app:app --reload
```
The API will be available at `http://127.0.0.1:8000`. Interactive docs are at `http://127.0.0.1:8000/docs`.

**Single prediction** — `POST /predict` (one image):

```json
{
  "prediction": "PNEUMONIA",
  "confidence": 0.9642,
  "prediction_index": 1
}
```

**Batch prediction** — `POST /predict/batch` (multiple images, form field `files`):

```json
{
  "count": 2,
  "results": [
    {
      "filename": "xray1.jpeg",
      "prediction": "PNEUMONIA",
      "confidence": 0.9642,
      "prediction_index": 1
    },
    {
      "filename": "xray2.jpeg",
      "prediction": "NORMAL",
      "confidence": 0.8123,
      "prediction_index": 0
    }
  ]
}
```

Example batch request with curl:

```bash
curl -X POST "http://127.0.0.1:8000/predict/batch" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg"
```

### 5. Run the Frontend App
In a separate terminal, start the Streamlit app:
```bash
streamlit run streamlit_app.py
```
The app will open in your browser.

## 🐳 Docker Deployment

Ensure ONNX model files exist in `modelWeights/` before building (see step 3 above).

### Run with Docker Compose (recommended)

```bash
docker compose up --build
```

| Service | URL |
|---------|-----|
| API | http://localhost:8000 |
| API docs | http://localhost:8000/docs |
| Streamlit UI | http://localhost:8501 |

Stop with `Ctrl+C` or `docker compose down`.

### Run services individually

```bash
# API only
docker build -f Dockerfile.api -t pneumonia-api .
docker run -p 8000:8000 pneumonia-api

# Streamlit only (API must already be running)
docker build -f Dockerfile.streamlit -t pneumonia-ui .
docker run -p 8501:8501 -e API_URL=http://host.docker.internal:8000 pneumonia-ui
```

## 🌐 Live Demo Deployment

> **Note:** `modelWeights/` is gitignored by default. For cloud deployment, commit the ONNX files (`pneumonia_model.onnx` + `pneumonia_model.onnx.data`) to your repo, use Git LFS, or attach them via your platform's storage before deploying.

### Option A — Render (API + UI)

1. Push this repo to GitHub (include ONNX weights or add them before deploy).
2. Go to [Render Dashboard](https://dashboard.render.com) → **New** → **Blueprint**.
3. Connect the repo — Render reads `render.yaml` and creates two services:
   - `pneumonia-api` — FastAPI backend
   - `pneumonia-ui` — Streamlit frontend (auto-linked via `API_URL`)
4. After deploy, your live links will be:
   - API: `https://pneumonia-api.onrender.com`
   - Demo UI: `https://pneumonia-ui.onrender.com`

Add the UI link to your resume and GitHub README:

```markdown
**Live Demo:** https://pneumonia-ui.onrender.com
```

### Option B — Hugging Face Spaces (UI) + Render (API)

1. Deploy the API on Render (steps above) or any public host.
2. Create a new [Streamlit Space](https://huggingface.co/new-space) from this GitHub repo.
3. Set **Settings → Repository secrets** → `API_URL` = your public API URL (e.g. `https://pneumonia-api.onrender.com`).
4. Use `streamlit_app.py` as the app entry point.

See `deploy/huggingface/README.md` for Space README template and setup notes.

### Environment variables

| Variable | Service | Default | Description |
|----------|---------|---------|-------------|
| `API_URL` | Streamlit | `http://127.0.0.1:8000` | FastAPI base URL |
| `ONNX_MODEL_PATH` | API | `modelWeights/pneumonia_model.onnx` | Path to ONNX model |
| `CORS_ORIGINS` | API | `*` | Allowed CORS origins (comma-separated) |

## 🧠 Model Implementation
The model utilizes **Transfer Learning** with a pre-trained **ResNet18** model.
- **Input**: $224 \times 224$ RGB images.
- **Modification**: The final fully connected layer is replaced with a linear layer outputting 2 classes.
- **Training**: The model was trained using CrossEntropyLoss and the Adam optimizer (PyTorch).
- **Inference**: The trained model is exported to ONNX and served with **ONNX Runtime** in the FastAPI backend. Softmax and confidence scoring run in NumPy after the ONNX forward pass.

### Inference pipeline
```text
Upload X-ray → preprocess (224×224, ImageNet normalize) → ONNX Runtime → softmax → prediction + confidence
```

## 📝 License
[Specify your license here, e.g., MIT]
