import os

import requests
import streamlit as st

API_BASE = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
ALLOWED_TYPES = ["jpg", "jpeg", "png"]
UPLOADER_LABEL = "Chest X-ray only (JPEG/PNG)"

st.title("Pneumonia Detection App")

st.warning(
    "**Medical disclaimer:** For educational and research use only. "
    "Not intended for clinical diagnosis. Always consult a qualified healthcare professional."
)

st.info("Please upload a **grayscale chest X-ray** image in JPEG or PNG format.")

tab_single, tab_batch = st.tabs(["Single Image", "Batch Upload"])

with tab_single:
    uploaded_file = st.file_uploader(
        UPLOADER_LABEL,
        type=ALLOWED_TYPES,
        key="single_upload",
        help="Chest X-ray images in JPEG or PNG format only.",
    )

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", width=300)

        if st.button("Predict Pneumonia"):
            with st.spinner("Predicting..."):
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type,
                    )
                }
                try:
                    response = requests.post(f"{API_BASE}/predict", files=files)
                    if response.status_code == 200:
                        result = response.json()
                        prediction = result.get("prediction", "Unknown")
                        confidence = result.get("confidence", 0)
                        confidence_pct = confidence * 100
                        if prediction == "PNEUMONIA":
                            st.error(f"Prediction: **{prediction}**")
                        else:
                            st.success(f"Prediction: **{prediction}**")
                        st.metric(label="Confidence", value=f"{confidence_pct:.1f}%")
                        st.progress(confidence)
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Connection error: {e}")

with tab_batch:
    uploaded_files = st.file_uploader(
        UPLOADER_LABEL,
        type=ALLOWED_TYPES,
        accept_multiple_files=True,
        key="batch_upload",
        help="Chest X-ray images in JPEG or PNG format only.",
    )

    if uploaded_files:
        st.write(f"{len(uploaded_files)} file(s) selected")
        cols = st.columns(min(len(uploaded_files), 4))
        for i, file in enumerate(uploaded_files[:4]):
            cols[i % 4].image(file, caption=file.name, width=150)
        if len(uploaded_files) > 4:
            st.caption(f"...and {len(uploaded_files) - 4} more")

        if st.button("Predict Batch"):
            with st.spinner(f"Predicting {len(uploaded_files)} images..."):
                files = [
                    ("files", (f.name, f.getvalue(), f.type))
                    for f in uploaded_files
                ]
                try:
                    response = requests.post(f"{API_BASE}/predict/batch", files=files)
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"Processed {data['count']} image(s)")
                        for item in data["results"]:
                            prediction = item["prediction"]
                            confidence_pct = item["confidence"] * 100
                            label = f"{item['filename']} — {prediction} ({confidence_pct:.1f}%)"
                            if prediction == "PNEUMONIA":
                                st.error(label)
                            else:
                                st.success(label)
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Connection error: {e}")
