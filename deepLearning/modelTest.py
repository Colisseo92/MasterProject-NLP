import os
import time

os.environ['TF_ENABLE_ONEDNN_OPTS']="0"

import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
import visualkeras

start_time = time.time()

from tensorflow.keras import layers

df = pd.read_csv("../datas/lemmatized2.csv",nrows=10000)

# Séparer les features (X) et la target (y)
X = df["lemmatized"]
y = df["Score"]
y = y.replace(1,0)
y = y.replace(2,0)
y = y.replace(3,0)
y = y.replace(4,1)
y = y.replace(5,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y,random_state=42)

# Tokenizer et séquences
max_words = 1000
max_len = 100

#tokenizer = Tokenizer(num_words=max_words)
#tokenizer.fit_on_texts(X_train)
vectorizer = layers.TextVectorization(max_tokens=max_words,output_sequence_length=max_len)
vectorizer.adapt(X_train)

#X_train_seq = tokenizer.texts_to_sequences(X_train)
#X_test_seq = tokenizer.texts_to_sequences(X_test)

#X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
#X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)
x_train = vectorizer(X_train)
x_test = vectorizer(X_test)

def base_model():

    model = Sequential()
    model.add(Embedding(input_dim=max_words, output_dim=128, input_length=max_len))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = Adam(learning_rate=0.001)

    model.compile(loss="mse",metrics=['accuracy'],optimizer=optimizer)
    return model

def train_and_evaluate():
    model = base_model()
    history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))
    loss, accuracy = model.evaluate(x_test, y_test)
    model.save("model.keras")
    visualkeras.layered_view(model)
    return history, accuracy

history, _ = train_and_evaluate()

#sns.set_style("darkgrid")
#plt.plot(history.history['accuracy'], label='accuracy')
#plt.plot(history.history['val_accuracy'], label='val_accuracy')
#plt.xlabel('Epoch')
#plt.ylabel('Accuracy')
#plt.title('Accuracy en fonction de l\'epoch [32] learning_rate=0,001')
#plt.legend()
#plt.show()

#plt.plot(history.history['loss'], label='loss')
#plt.plot(history.history['val_loss'], label='val_loss')
#plt.xlabel('Epoch')
#plt.ylabel('loss')
#plt.title('Loss en fonction de l\'epoch [64,32] learning_rate=0,001')
#plt.legend()
#plt.show()

end_time = time.time()
print(f"time to run : {end_time - start_time}s")