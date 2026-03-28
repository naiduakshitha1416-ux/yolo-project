import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.title("PAN Card Detection")

model = YOLO("best.pt")
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    results = model(image)
    result_img = results[0].plot()

    st.image(result_img, caption="Detected Output")