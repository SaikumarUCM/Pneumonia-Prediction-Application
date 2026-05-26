import streamlit as st
import requests

st.title("Pneumonia Detection App")

st.write("Upload a chest X-ray image to predict if it shows pneumonia.")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", width=300)

    if st.button("Predict Pneumonia"):
        with st.spinner("Predicting..."):
            files = {"file": uploaded_file.getvalue()}
            try:
                response = requests.post("http://127.0.0.1:8000/predict", files=files)
                if response.status_code == 200:
                    result = response.json()
                    prediction = result.get("prediction", "Unknown")
                    if prediction == "PNEUMONIA":
                        st.error(f"Prediction: {prediction}")
                    else:
                        st.success(f"Prediction: {prediction}")
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Connection error: {e}")