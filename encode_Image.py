import pickle
import tensorflow as tf
import keras
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import (load_img, img_to_array)
from tensorflow import ConfigProto
import numpy as np


def encode_images(img_name):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    keras.backend.tensorflow_backend.set_session(tf.Session(config=config))
    image_model = ResNet50(include_top=False, weights='imagenet', pooling='avg')

    encoding = {}
    print('encoding test images')
    image_input = np.empty((1, 224, 224, 3))
    img = img_name
    img = load_img(img, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = keras.applications.resnet50.preprocess_input(img_array)
    image_input[0] = img_array

    preds = image_model.predict(image_input)
    encoding[img_name] = preds

    return encoding
 #   filename = 'input_test.p'
 #  with open(filename, 'wb') as encoded_pickle:
 #       pickle.dump(encoding, encoded_pickle)
