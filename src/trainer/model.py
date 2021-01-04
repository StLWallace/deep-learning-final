"""
This contains the model architecture. 
Changes to the architecture here will be propogated to the training task

Args:
    glove_layer - an optional glove embedding layer. If this remains none, a stock embedding layer will be used
    max_nb_words - the max number of words to be used in the embedding layers if glove_layer is None
    max_seq_length - also used in embedding layer if glove not supplied
"""
from keras.layers import Embedding, LSTM, SimpleRNN, Dense, Dropout
from keras.models import Sequential


def get_model(glove_layer=None, max_nb_words=10000, max_sequence_length=1000):
    if glove_layer is None:
        embedding_layer = Embedding(max_nb_words, 64, input_length=max_sequence_length)
    else:
        embedding_layer = glove_layer
    model = Sequential()
    model.add(embedding_layer)
    model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(
        loss='binary_crossentropy',
        optimizer='adamax',
        metrics=['acc'])
    
    return model