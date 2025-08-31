from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import models
import streamlit as st
import cv2
import numpy as np
from img_class import import_and_predict


st.write("""
    # 10 Class Desserts Classifier
"""
)

st.write("""This is a simple image classification webapp to predict 10 classes of desserts: 
        baklava, bread pudding, carrot cake, cheese cake, cupcakes, chocolate cake, tiramisu, 
        red velvet cake, strawberry shortcake and creme brulee. The model was trained on EfficientNetB0
        and has achieved 77% validation accuracy.  
""")

st.write("Do note that the classifier is not 100% accurate and may tend to misclassify certain images like carrot cake with cheesecake etc.")

file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, './finalmodel/final_model.h5')
    
    if np.argmax(prediction) == 0:
        st.write("It is baklava!")
    elif np.argmax(prediction) == 1:
        st.write("It is bread pudding!")
    elif np.argmax(prediction) == 2:
        st.write("It is carrot cake!")
    elif np.argmax(prediction) == 3:
        st.write("It is cheesecake!")
    elif np.argmax(prediction) == 4:
        st.write("It is chocolate cake!")
    elif np.argmax(prediction) == 5:
        st.write("It is creme brulee!")
    elif np.argmax(prediction) == 6:
        st.write("It is cupcake!")
    elif np.argmax(prediction) == 7:
        st.write("It is red velvet cake!")
    elif np.argmax(prediction) == 8:
        st.write("It is strawberry shortcake!")
    elif np.argmax(prediction) == 9:
        st.write("It is tiramisu!")
    else:
        st.write("It does not look like yummy dessert!")
    

