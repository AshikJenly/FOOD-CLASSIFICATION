import streamlit as st
from PIL import Image
from model import get_model, process_image, get_prediction
import os
import tempfile

st.title("Food Classification")
st.sidebar.title("Navigation")
selected_model = st.sidebar.radio("Select Model", ["Efficent Net 0", "Efficient Net 1", "Efficient Net 2"])
model = get_model(selected_model)

temp_dir = tempfile.TemporaryDirectory()
temp_dir_path = temp_dir.name

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_path = os.path.join(temp_dir_path, uploaded_file.name)
    with open(file_path, 'wb') as file:
        file.write(uploaded_file.read())

    image = Image.open(uploaded_file)
    
    # Process the image
    actual_image = process_image(image_path=file_path)[0]

    # Get prediction
    prediction = get_prediction(model, actual_image)
    st.markdown(f"<div style='display:flex'><h3>Prediced as : </h3><h1>{prediction}</h1></div>",unsafe_allow_html=True)
    st.image(image, caption="Uploaded Image", use_column_width=True, width=300)

 