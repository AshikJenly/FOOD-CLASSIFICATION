import streamlit as st
from PIL import Image
from model import get_model, process_image, get_prediction
import os
import tempfile

SAMPLE_IMAGE_DIR = "sample_images"

def get_sample_image_paths():
    return [img for img in os.listdir(SAMPLE_IMAGE_DIR)]

st.title("Food Classification")
st.write("""
            This is Food prediction webapp 
         """)
st.sidebar.title("Navigation")

selected_model = st.sidebar.radio("Select Model", ["Efficient Net 0", "Efficient Net 1", "Efficient Net 2"],2)
model = get_model(selected_model)

# Get sample image paths
sample_image_paths = get_sample_image_paths()
sample_image_paths.append("Upload Your Own Image") 
selected_image_option = st.sidebar.selectbox("Select Image", sample_image_paths)

if selected_image_option == "Upload Your Own Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        temp_dir = tempfile.TemporaryDirectory()
        temp_dir_path = temp_dir.name
        file_path = os.path.join(temp_dir_path, uploaded_file.name)
        with open(file_path, 'wb') as file:
            file.write(uploaded_file.read())

        image = Image.open(uploaded_file)


        actual_image = process_image(image_path=file_path)[0]


        prediction = get_prediction(model, actual_image)
        st.markdown(f"<div style='display:flex'><h3>Predicted as : </h3><h1>{prediction}</h1></div>", unsafe_allow_html=True)
        st.image(image, caption="Uploaded Image", use_column_width=True, width=300)

else:
    
    image_path = os.path.join(SAMPLE_IMAGE_DIR, selected_image_option)
    image = Image.open(image_path)
    
    actual_image = process_image(image_path=image_path)[0]

    prediction = get_prediction(model, actual_image)
    
    st.markdown(f"<div style='display:flex'><h3>Predicted as : </h3><h1>{prediction}</h1></div>", unsafe_allow_html=True)
    st.image(image, caption="Sample Image", use_column_width=True, width=300)
