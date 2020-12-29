import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding
import os
import numpy as np

# Global Constants
DF_PATH = "gs://slalom-stl-kaggle-datasets/fake-comments"
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
GLOVE_PATH = "/home/jupyter/final/glove/glove.6B.100d.txt"

## This class loads and
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

def process():
    ## Initialize ModelData object
    md = ModelData(df_path=DF_PATH).split_data().get_tokenizer()
    # Get datasets
    X_train, y_train, X_test, y_test = md.return_all_data()
    # define word_index
    word_index = md.get_word_index()

    ## Get GloVe embedding
    glove = MakeGlove(glove_path=GLOVE_PATH, word_index=word_index).get_embedding_layer()


kf = StratifiedKFold(n_splits=6)

for train_index, test_index in kf.split(X_train, y_train):
    print("TRAIN:", train_index, "TEST:", test_index)

