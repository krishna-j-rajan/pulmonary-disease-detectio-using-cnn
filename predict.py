from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.utils import normalize
import numpy as np

IMG_SIZE = 224
class_names = ['Normal', 'Bacteria-Pneumonia', 'Virus-Pneumonia']
model = keras.models.load_model("./pneumonia_train.h5")


def getPrediction(filename):
    img = keras.preprocessing.image.load_img(
        filename, target_size=(IMG_SIZE, IMG_SIZE)
    )
    img_array = np.array(img).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    img_array = normalize(img_array, axis=1)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    return ({"class_num": np.argmax(score), "class_name": class_names[np.argmax(score)]})
