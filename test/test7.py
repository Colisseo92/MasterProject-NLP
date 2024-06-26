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

start_time = time.time()

from tensorflow.keras import layers

df = pd.read_csv("lemmatized3.csv")
df.dropna(subset=["lemmatized", "Score"], inplace=True)

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

model_vec = Sequential()
model_vec.add(tf.keras.Input(shape=(1,),dtype=tf.string))
model_vec.add(vectorizer)

model_vec.save('tf-vectorizer.keras')

print("saved")

#X_train_seq = tokenizer.texts_to_sequences(X_train)
#X_test_seq = tokenizer.texts_to_sequences(X_test)

#X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
#X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)
x_train = vectorizer(X_train)
x_test = vectorizer(X_test)

embedded_layer = Embedding(
    1000,
    128,
    trainable=False
)
embedded_layer.build((1,))

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

model = base_model()
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))


predict = model.predict()

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