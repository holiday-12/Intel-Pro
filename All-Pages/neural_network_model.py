import os
import numpy as np
from PIL import Image
import cv2
import streamlit as st
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.applications.vgg19 import VGG19

# Load the model
base_model = VGG19(include_top=False, input_shape=(240, 240, 3))
x = base_model.output
flat = Flatten()(x)
class_1 = Dense(4608, activation='relu')(flat)
drop_out = Dropout(0.2)(class_1)
class_2 = Dense(1152, activation='relu')(drop_out)
output = Dense(2, activation='softmax')(class_2)
model_03 = Model(base_model.inputs, output)
model_03.load_weights('All-Model/NN/vgg19_unfrozen.h5')

# Print message
st.write('Model loaded. Please upload an image.')

# Function to get class name based on predicted class number
def get_className(classNo):
    if classNo == 0:
        return "No Brain Tumor"
    elif classNo == 1:
        return "Yes Brain Tumor"

# Function to get prediction result
def getResult(img):
    image = cv2.imread(img)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((240, 240))
    image = np.array(image)
    input_img = np.expand_dims(image, axis=0)
    result = model_03.predict(input_img)
    result01 = np.argmax(result, axis=1)
    return result01

# Streamlit UI
st.title("Brain Tumor Classification Using Deep Learning")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

# Show image preview if uploaded
if uploaded_file is not None:
    # Save file to disk
    img_path = os.path.join("uploads", uploaded_file.name)
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Button to predict
    if st.button("Predict"):
        # Get result
        value = getResult(img_path)
        result = get_className(value)
        st.write(f"Prediction: {result}")
