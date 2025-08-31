from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import streamlit as st

# Load model once at startup
model = tf.keras.models.load_model('./finalmodel/final_model.h5')

# Updated import_and_predict function
def import_and_predict(image_data):
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img_array = np.array(image) / 255.0  # scale pixels
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
    prediction = model.predict(img_array)
    return prediction

st.write("""
    # 10 Class Desserts Classifier
""")

st.write("""This is a simple image classification webapp to predict 10 classes of desserts: 
        baklava, bread pudding, carrot cake, cheesecake, cupcakes, chocolate cake, tiramisu, 
        red velvet cake, strawberry shortcake, and creme brulee. The model was trained on EfficientNetB0
        and has achieved 77% validation accuracy.
""")

st.write("Do note that the classifier is not 100% accurate and may misclassify certain images.")

file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    
    # Call the updated function (no model path argument)
    prediction = import_and_predict(image)
    
    # Map class indices to names
    classes = [
        "baklava", "bread pudding", "carrot cake", "cheesecake", 
        "chocolate cake", "creme brulee", "cupcake", "red velvet cake", 
        "strawberry shortcake", "tiramisu"
    ]
    
    st.write(f"It is {classes[np.argmax(prediction)]}!")
