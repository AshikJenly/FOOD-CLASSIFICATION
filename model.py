
import tensorflow as tf
import numpy as np
import json

json_file_path = './data/indices.json'
DATA = {}
with open(json_file_path, 'r') as json_file:
    DATA = json.load(json_file)


def get_model(model_name):
    model = None
    if model_name == "Efficient Net 0":
        model = tf.keras.models.load_model("./data/food101_model_0.h5")
    elif model_name == "Efficient Net 1":
        model = tf.keras.models.load_model("./data/food101_model_1.h5")
    elif model_name == "Efficient Net 2":
        model = tf.keras.models.load_model("./data/food101_model_2a.h5")

    return model
def process_image(image_path):
    
    image_raw = tf.io.read_file(image_path)
    image = tf.image.decode_image(image_raw, channels=3)
    image_resized = tf.image.resize(image, (224,224))
    image_resized = tf.expand_dims(image_resized,axis=0)
    
    valid = True
    return image_resized,valid

def get_prediction(model,image):

    pred = model.predict(image)
    return list(DATA)[np.argmax(pred)]
    
