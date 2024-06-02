import os
import time

os.environ['TF_ENABLE_ONEDNN_OPTS']="0"

import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Embedding, LSTM, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.preprocessing.text import Tokenizer
from keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
start_time = time.time()

df = pd.read_csv("../datas/lemmatized2.csv")

# Séparer les features (X) et la target (y)
X = df["lemmatized"]
y = df["Score"]
y = y.replace(1,0)
y = y.replace(2,0)
y = y.replace(3,0)
y = y.replace(4,1)
y = y.replace(5,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y,random_state=42)


max_words = 1000
max_len = 100

vectorizer = layers.TextVectorization(max_tokens=max_words,output_sequence_length=max_len)
vectorizer.adapt(X_train)

voc = vectorizer.get_vocabulary()
word_index = dict(zip(voc,range(len(voc))))

path_to_glove = "../datas/glove.6B.100d.txt"

embeddings_index = {}
with open(path_to_glove,encoding="utf-8") as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs,"f",sep=" ")
        embeddings_index[word] = coefs

print("found %s word vectors" % len(embeddings_index))

num_token = len(voc) + 2
embedding_dim = 100
hits = 0
misses = 0

#prepare embedding matrix
embedding_matrix = np.zeros((num_token,embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
        hits += 1
    else:
        misses += 1

print("Converted %d words (%d misses)" % (hits,misses))

embedded_layer = Embedding(
    num_token,
    embedding_dim,
    trainable=False
)
embedded_layer.build((1,))
embedded_layer.set_weights([embedding_matrix])

def base_model():

    model = Sequential()
    model.add(embedded_layer)
    model.add(GlobalAveragePooling1D())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = Adam(learning_rate=0.001)

    model.compile(loss="mse",metrics=['accuracy'],optimizer=optimizer)
    return model

x_train = vectorizer(X_train)
x_test = vectorizer(X_test)

def train_and_evaluate():
    model = base_model()
    history = model.fit(x_train, y_train, epochs=20, batch_size=128, validation_data=(x_test, y_test))
    loss, accuracy = model.evaluate(x_test, y_test)
    return history, accuracy

history, _ = train_and_evaluate()


end_time = time.time()
print(f"time to run : {end_time - start_time}s")