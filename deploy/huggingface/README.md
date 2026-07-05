---
title: Pneumonia Detection
emoji: 🩻
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.57.0"
app_file: streamlit_app.py
pinned: false
license: mit
---

# Pneumonia Detection (Streamlit UI)

Upload chest X-ray images for pneumonia classification.

## Setup

This Space is the **frontend only**. It calls a deployed FastAPI backend for ONNX inference.

1. Deploy the API first (Docker on Render, or run locally).
2. In this Space, go to **Settings → Repository secrets** and add:

| Secret | Example |
|--------|---------|
| `API_URL` | `https://pneumonia-api.onrender.com` |

3. Restart the Space.

## Local + Docker

See the main repository README for `docker compose up --build`.

## Disclaimer

For educational and research use only. Not intended for clinical diagnosis.
