from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input
import tensorflow as tf

# Load the model once at the top of this file
model = tf.keras.models.load_model('./finalmodel/final_model.h5')

def import_and_predict(image_data):
    size = (224, 224)
    image_data = image_data.convert("RGB")  # ensure 3 channels

    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        resample = Image.ANTIALIAS

    image = ImageOps.fit(image_data, size, resample)
    img_array = np.array(image)
    img_array = preprocess_input(img_array)  # EfficientNet preprocessing
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    return prediction
