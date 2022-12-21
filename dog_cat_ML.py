import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import tensorflow_hub as hub
mobilenet_model = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4'
path = 'dog_cat.h5'
data= tf.keras.models.load_model(path ,custom_objects={'KerasLayer': hub.KerasLayer})
img_path=st.text_input("Enter the path of the image: ")
img=Image.open(img_path)
st.image(img, caption='Image of given Path')
if(st.button("Predict Result")):

    input_image = Image.open(img_path)
    input_image = input_image.resize((224, 224))
    input_image_scaled = input_image / 255

    image_reshaped = np.reshape(input_image_scaled, [1, 224, 224, 3])

    input_prediction = data.predict(image_reshaped)


    input_pred_label = np.argmax(input_prediction)


    if input_pred_label == 0:
        st.success('The image represents a Cat')

    else:
        st.success('The image represents a Dog')
