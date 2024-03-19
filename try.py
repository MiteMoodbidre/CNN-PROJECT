import streamlit as st
from PIL import Image
import numpy as np
from keras.models import load_model

# Load the models (replace these with your actual model paths)
inception_model = load_model('inceptionv3_trained_model.h5')
xception_model = load_model('xception_trained_model.h5')
mobilenet_model = load_model('mobilenet_trained_model.h5')

# Function to preprocess images for each model
def preprocess_image(image, model_name):
    target_size = (299, 299) if model_name == 'inception' or model_name == 'xception' else (224, 224)
    image = image.resize(target_size)
    image = np.expand_dims(image, axis=0)
    return image

# Title and custom background
st.title("Bone Marrow Cancer Detection")
st.markdown(
    """
    <style>
    body {
        background-color: #f0f2f6;  /* Light background color */
        color: #000000;  /* Black text color */
    }
    .doctor-image {
        width: 100px;  /* Adjust the width as needed */
        height: 100px;  /* Adjust the height as needed */
        border-radius: 50%;  /* Circular shape */
        box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.2);  /* Box shadow for a subtle effect */
        margin: auto;  /* Center align the image */
        display: block;  /* Make image a block element for margin:auto to work */
    }
    .prediction-message {
        font-size: 18px;  /* Adjust the font size as needed */
        margin-top: 20px;  /* Space between image and message */
        text-align: center;  /* Center align the message */
    }
    .benign-message {
        color: green;  /* Green color for benign prediction */
    }
    .malignant-message {
        color: red;  /* Red color for malignant prediction */
    }
    </style>
    """
    ,
    unsafe_allow_html=True
)

# Display image below the heading
image_below_heading = Image.open('bg.jpg')  # Replace 'bg.jpg' with the actual image path
st.image(image_below_heading, caption='Dr. Smith', use_column_width=True)

# Description of the project
st.markdown("""
    This project aims to detect bone marrow cancer using deep learning models. Please upload an image below to make predictions.
""")

# Display image and checkbox
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg"])

# Show prediction message only when image is uploaded
if uploaded_file is not None:
    # Load and preprocess the uploaded image
    image = Image.open(uploaded_file)
    # st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image for each model
    inception_input = preprocess_image(image.copy(), 'inception')
    xception_input = preprocess_image(image.copy(), 'xception')
    mobilenet_input = preprocess_image(image.copy(), 'mobilenet')

    # Predictions
    inception_pred = inception_model.predict(inception_input)
    xception_pred = xception_model.predict(xception_input)
    mobilenet_pred = mobilenet_model.predict(mobilenet_input)

    # Determine the predicted class for each model
    benign_count = 0
    malignant_count = 0

    if inception_pred[0][0] < 0.5:
        benign_count += 1
    else:
        malignant_count += 1

    if xception_pred[0][0] < 0.5:
        benign_count += 1
    else:
        malignant_count += 1

    if mobilenet_pred[0][0] < 0.5:
        benign_count += 1
    else:
        malignant_count += 1

    # Predict the final class based on the counts
    prediction_message = ""
    if benign_count > malignant_count:
        prediction_message = "Predicted Class: Benign"
        st.markdown(f'<p class="prediction-message benign-message">{prediction_message}</p>', unsafe_allow_html=True)
    else:
        prediction_message = "Predicted Class: Malignant"
        st.markdown(f'<p class="prediction-message malignant-message">{prediction_message}</p>', unsafe_allow_html=True)
