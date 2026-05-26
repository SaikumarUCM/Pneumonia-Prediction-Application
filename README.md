# Pneumonia Prediction App

An AI-powered application designed to detect pneumonia from chest X-ray images using deep learning. This project implements a complete pipeline from model training to a deployable web application.

## 🚀 Features
- **Deep Learning Model**: Uses a fine-tuned ResNet18 architecture for high-accuracy binary classification (Normal vs. Pneumonia).
- **FastAPI Backend**: A robust REST API that handles image uploads and provides predictions.
- **Streamlit Frontend**: A user-friendly web interface for uploading X-ray images and viewing real-time predictions.
- **Pre-processing Pipeline**: Integrated image normalization and resizing to ensure consistent model input.

## 📸 Screenshots

![App Overview](utils/screenshot-1.png)

![Detailed View](utils/screenshot-2.png)

## 🛠️ Tech Stack
- **Language**: Python
- **Deep Learning Framework**: PyTorch, Torchvision
- **Backend Framework**: FastAPI
- **Frontend Framework**: Streamlit
- **Image Processing**: PIL (Pillow)
- **Metrics**: Scikit-learn (Accuracy Score)

## 📂 Project Structure
```text
.
├── Model Notebooks/       # Training scripts and dataset
│   ├── data/              # Chest X-ray dataset (train, test, val)
│   └── train_model.py     # Model training and evaluation script
├── modelWeights/          # Saved trained model weights (.pth)
├── src/                   # Source code
│   ├── api/               # FastAPI implementation
│   │   └── app.py         # API endpoints and model loading
│   ├── core/              # Core utility functions
│   │   └── utils.py       # Image preprocessing logic
│   └── pneumonia_model.py # Model architecture and loading logic
├── streamlit_app.py       # Streamlit frontend application
├── requirements.txt       # Project dependencies
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

### 3. Run the Backend API
Start the FastAPI server:
```bash
uvicorn src.api.app:app --reload
```
The API will be available at `http://127.0.0.1:8000`.

### 4. Run the Frontend App
In a separate terminal, start the Streamlit app:
```bash
streamlit run streamlit_app.py
```
The app will open in your browser.

## 🧠 Model Implementation
The model utilizes **Transfer Learning** with a pre-trained **ResNet18** model. 
- **Input**: $224 \times 224$ RGB images.
- **Modification**: The final fully connected layer is replaced with a linear layer outputting 2 classes.
- **Training**: The model was trained using CrossEntropyLoss and the Adam optimizer.

## 📝 License
[Specify your license here, e.g., MIT]
