import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding, LSTM, SimpleRNN, Dense, Dropout
from keras.models import Sequential
from keras.metrics import AUC
import os
import numpy as np
from google.cloud import storage
from io import StringIO
import argparse
import pickle
from tensorflow.python.lib.io import file_io
from tensorflow.io.gfile import GFile
import tensorflow as tf

""" Trying to get TPU stuff to work. Maybe later
resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.TPUStrategy(resolver)
"""
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

"""
Global Constants - set these to your values to run
"""
# This is the path to your local folder containing True.csv and Fake.csv
DF_PATH = "gs://slalom-stl-kaggle-datasets/fake-comments"
OUTPUT_PATH = "gs://slalom-stl-kaggle-datasets/fake-comments/results"
# Whatever you want these to be
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 1000
# Path to your local (unzipped) glove.6B.100d.txt
GLOVE_PATH = "/home/jupyter/final/glove/glove.6B.100d.txt"

## This class loads and cleans data
class ModelData():
    
    def __init__(self, df_path):
        self.train_df = None
        self.test_df = None
        self.tokenizer = None
        self.df_path = df_path
    
    def get_df(self, df_path=None):
        if df_path is None:
            df_path = self.df_path
        # Read datasets from Cloud Storage
        true_df = pd.read_csv(df_path+"/True.csv")
        fake_df = pd.read_csv(df_path+"/Fake.csv")

        # Add labels
        true_df["true"] = 1
        fake_df["true"] = 0

        # Combine dfs
        trueFake_df = pd.concat([true_df, fake_df], axis=0)

        # Add text length field
        trueFake_df['textLength'] = trueFake_df.text.apply(lambda x: len(x) if x is not None and len(x) > 0 else 0)

        # Filter out obs with 0 lenth text
        trueFake_df = trueFake_df.loc[trueFake_df['textLength'] > 0]
        trueFake_df.reset_index(inplace=True, drop=True)

        return trueFake_df
    
    def split_data(self):
        df = self.get_df()
        self.train_df, self.test_df = train_test_split(df, train_size=0.8)
        return self
    
    def get_tokenizer(self, max_nb_words=20000):    
        if not any([self.train_df is None, self.test_df is None]):
            train_texts = list(self.train_df["text"])
            tokenizer = Tokenizer(num_words=max_nb_words)
            # Tokenizer should be fit on train data
            tokenizer.fit_on_texts(train_texts)
            self.tokenizer = tokenizer
        return self
    
    def get_encoded_data(self, df, max_sequence_length=1000):
        texts = list(df["text"])
        labels = list(df["true"])
        sequences = self.tokenizer.texts_to_sequences(texts)

        data = pad_sequences(sequences, maxlen=max_sequence_length)
        labels = np.asarray(labels)
            
        return data, labels
    
    def get_word_index(self):
        word_index = self.tokenizer.word_index
        return word_index
        
    def return_all_data(self):
        X_train, y_train = self.get_encoded_data(self.train_df)
        X_test, y_test = self.get_encoded_data(self.test_df)
        return X_train, y_train, X_test, y_test
    

## This class will be used to return the GloVe embedding layer
# Required args:
# glove_path - path to glove files
# word_index - a word_index object as created in ModelData().get_word_index()
class MakeGlove():
    
    def __init__(self, glove_path, word_index, embedding_dim=100, max_sequence_length=1000):
        self.glove_path = glove_path
        self.word_index = word_index
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length

    def get_embedding_index(self, glove_path=None):
        if glove_path is None:
            glove_path = self.glove_path
        embeddings_index = {}
        # Use this to read from Cloud Storage
        if "gs://" in glove_path:
            f = GFile(glove_path, mode='r')
        else:
            f = open(glove_path)
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        return embeddings_index
    
    def get_matrix(self):
        embedding_matrix = np.zeros((len(self.word_index) + 1, self.embedding_dim))
        embeddings_index = self.get_embedding_index()
        for word, i in self.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        return embedding_matrix
    
    def get_embedding_layer(self):
        embedding_layer = Embedding(
            len(self.word_index) + 1,
            self.embedding_dim,
            weights=[self.get_matrix()],
            input_length=self.max_sequence_length,
            trainable=False)
        return embedding_layer

    
## Use this class to fit a model using Kfold cross validation
class FitModel():
    
    def __init__(self, model, X_train, y_train, X_test, y_test):
        self.kf = StratifiedKFold(n_splits=6)
        self.results = []
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model = model
        
    def fit_model(self, model, X_train, y_train, X_val, y_val, batch_size=256):
        history = model.fit(
            X_train, 
            y_train,
            epochs=10,
            batch_size=batch_size,
            validation_data=(X_val, y_val))

        return history
    
    # Using 6-fold cross validation
    def k_fold_cv(self):
        model = self.model
        kf = StratifiedKFold(n_splits=6)
        i = 1
        for train_index, test_index in kf.split(self.X_train, self.y_train):
            Xt = self.X_train[train_index]
            yt = self.y_train[train_index]
            Xv = self.X_train[test_index]
            yv = self.y_train[test_index]
            res = self.fit_model(model, Xt, yt, Xv, yv)
            self.results.append(res)
            print("######## Fold #", i)
            i += 1
        return self
        

## Run the whole thing
# If you want to run this interactively, you can just use the inside of the function
def process(glove_path=GLOVE_PATH, output_path=OUTPUT_PATH):
    ## Initialize ModelData object
    md = ModelData(df_path=DF_PATH).split_data().get_tokenizer(max_nb_words=MAX_NB_WORDS)
    # Get datasets
    X_train, y_train, X_test, y_test = md.return_all_data()
    print("Successfully loaded and transformed test and train data")
    # define word_index
    word_index = md.get_word_index()
    print("word index created")

    ## Get GloVe embedding
    glove = MakeGlove(glove_path=glove_path, word_index=word_index).get_embedding_layer()
    print("GloVe embedding defined")

    # This is the actual model you want to fit
    # haven't made any automated way to iterate through architecture
    with strategy.scope():
        model = Sequential()
        model.add(glove)
        #model.add(Embedding(MAX_NB_WORDS, 64, input_length=MAX_SEQUENCE_LENGTH))
        model.add(LSTM(32,dropout=0.2,recurrent_dropout=0.2))
        model.add(Dense(1))
        model.compile(
            loss='binary_crossentropy',
            optimizer='adamax',
            metrics=['acc'])

    mod_res = FitModel(model, X_train, y_train, X_test, y_test)
    mod_res.k_fold_cv()
    
    # Write to cloud storage
    with file_io.FileIO(output_path, mode='rb') as f:
        pickle.dump(mod_res.results, f, pickle.HIGHEST_PROTOCOL)
    
    with file_io.FileIO(output_path+"/model", mode='rb') as f2:
        pickle.dump(mod_res.model, f2, pickle.HIGHEST_PROTOCOL)

## For running from command line
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--glove_path')
    parser.add_argument('--output_path')
    parser.add_argument('--job-dir')
    args = parser.parse_args()
    process(glove_path=args.glove_path, output_path=args.output_path)
