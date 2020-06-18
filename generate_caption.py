import os
import pickle
import keras.backend as K
from config import best_model
from model import build_model
from tensorflow import ConfigProto
import keras
import tensorflow as tf


def generate(encoded_dic):

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    keras.backend.tensorflow_backend.set_session(tf.Session(config=config))


    model_weights_path = os.path.join('models', best_model)
    model = build_model()
    model.load_weights(model_weights_path)

    vocab = pickle.load(open('data/vocab_train.p', 'rb'))
    idx2word = sorted(vocab)
    word2idx = dict(zip(idx2word, range(len(vocab))))

    # encoded_test_a = pickle.load(open('input_test.p', 'rb'))

    names = [f for f in encoded_dic.keys()]

    sentences = []
    for i in range(len(names)):
        image_name = names[i]

        from beam_search import beam_search_predictions

        candidate = beam_search_predictions(model, image_name, word2idx,
                                            idx2word, encoded_dic, beam_index=20)
        sentences.append(candidate)

    K.clear_session()

    return ''.join(sentences)


if __name__ == '__main__':

    txt = generate()
    print(txt)