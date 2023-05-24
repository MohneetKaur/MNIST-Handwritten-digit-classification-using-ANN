import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

# Load the pre-trained model
model = load_model('model.h5')

# Set the page title
st.title("Digit Recognizer App")

# Upload and preprocess the image
uploaded_image = st.file_uploader("Upload an image", type=['png', 'jpg','jpeg'])
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    img_array = np.array(image.convert('L'))

    # Check image dimensions
    if len(img_array.shape) == 2:
        # Add extra dimensions for channel and batch size
        img_array = np.expand_dims(img_array, axis=(0, -1))
    elif len(img_array.shape) == 3:
        # Add extra dimension for batch size
        img_array = np.expand_dims(img_array, axis=0)
    else:
        st.error("Invalid image dimensions. Please upload a valid image.")

    resized_img = tf.image.resize(img_array, [28, 28])
    normalized_img = resized_img / 255.0
    reshaped_img = np.reshape(normalized_img, (1, 28, 28, 1))

    # Make predictions
    predictions = model.predict(reshaped_img)
    predicted_class = np.argmax(predictions)

    st.write(f"Predicted Digit: {predicted_class}")
