from PIL import Image
import streamlit as st
import numpy as np
from img_class import import_and_predict  # make sure img_class.py has the fixed function

# --- Streamlit App ---
st.title("🍰 10-Class Desserts Classifier")

st.write("""
This webapp predicts 10 classes of desserts:
- baklava
- bread pudding
- carrot cake
- cheesecake
- cupcakes
- chocolate cake
- tiramisu
- red velvet cake
- strawberry shortcake
- creme brulee

The model was trained on EfficientNetB0 and has ~77% validation accuracy.
Note: predictions are not 100% accurate.
""")

# File uploader
file = st.file_uploader("Upload an image file", type=["jpg", "png"])

if file is None:
    st.text("Please upload an image file to get a prediction.")
else:
    image = Image.open(file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # --- Make prediction ---
    prediction = import_and_predict(image)

    # Map class index to class name
    classes = [
        "baklava", "bread pudding", "carrot cake", "cheesecake",
        "chocolate cake", "creme brulee", "cupcake", "red velvet cake",
        "strawberry shortcake", "tiramisu"
    ]

    predicted_class = classes[np.argmax(prediction)]
    st.success(f"Predicted Dessert: **{predicted_class}**")
