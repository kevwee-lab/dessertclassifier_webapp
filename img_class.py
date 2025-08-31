from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import streamlit as st

# Load the model once at app startup
model = tf.keras.models.load_model('./finalmodel/final_model.h5')

def import_and_predict(image_data):
    # Resize and preprocess image
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img_array = np.array(image) / 255.0  # scale pixels
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
    
    # Predict
    prediction = model.predict(img_array)
    return prediction
