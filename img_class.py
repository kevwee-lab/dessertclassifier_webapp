import keras
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
from tensorflow import keras
import streamlit as st
from keras import backend as K
from cv2 import cv2
import numpy as np

@st.cache(allow_output_mutation=True)
def import_and_predict(image_data, model):
        model = tf.keras.models.load_model(model)
        size = (224,224)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = (cv2.resize(img, dsize=(224, 224),  interpolation=cv2.INTER_CUBIC))
        
        img_reshape = img_resize[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        
        return prediction

        