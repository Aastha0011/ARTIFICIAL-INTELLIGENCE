import streamlit as st
import joblib
import numpy as np
from PIL import Image

# Load model
model = joblib.load("PROJECT 8 MALE AND FEMALE IMAGE CLASSIFIER/FEMALE_MALE_model.pkl")

IMG_SIZE = 64

st.set_page_config(page_title="Gender Classification", page_icon="🧑")

st.title("🧑 Male/Female Image Classifier")

st.write("Upload an image to predict the gender.")

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert to RGB
    image = image.convert("RGB")

    # Resize
    image = image.resize((IMG_SIZE, IMG_SIZE))

    # Convert to numpy array
    img = np.array(image)

    # Flatten
    img = img.flatten()

    # Reshape for prediction
    img = img.reshape(1, -1)

    # Predict
    prediction = model.predict(img)[0]

    # Display result
    if prediction == 0:
        st.success("Prediction: Female 👩")
    else:
        st.success("Prediction: Male 👨")
