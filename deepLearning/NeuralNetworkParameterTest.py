import os

os.environ['TF_ENABLE_ONEDNN_OPTS']="0"

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import seaborn as sns

# Charger les données
df = pd.read_csv("../datas/lemmatized3.csv")
df.dropna(subset=["lemmatized", "Score"], inplace=True)

# Séparer les features (X) et la target (y)
X = df["lemmatized"]
y = df["Score"]
y = y.replace(1,0)
y = y.replace(2,0)
y = y.replace(3,0)
y = y.replace(4,1)
y = y.replace(5,1)


# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y, random_state=42)

# Tokenizer et séquences
max_words = 1000
max_len = 100

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

# Fonction pour créer et compiler le modèle avec une couche d'embedding
def create_model(learning_rate=0.001, optimizer='adam', layers=[64, 32], embedding_dim=128):
    model = Sequential()
    model.add(Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len))
    model.add(GlobalAveragePooling1D())
    for neurons in layers:
        model.add(Dense(neurons, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Utiliser 'sigmoid' pour une classification binaire

    # Choisir l'optimiseur
    if optimizer == 'adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        opt = SGD(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        opt = RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError("Optimizer not recognized")

    model.compile(optimizer=opt, loss='mse', metrics=['accuracy'])  # Utiliser 'binary_crossentropy' pour une cible binaire
    return model

# Entraîner et évaluer le modèle avec différentes configurations
def train_and_evaluate(learning_rate, optimizer, layers):
    model = create_model(learning_rate, optimizer, layers)
    history = model.fit(X_train_pad, y_train, epochs=10, batch_size=32, validation_data=(X_test_pad, y_test))
    loss, accuracy = model.evaluate(X_test_pad, y_test)
    return history, accuracy

# Configurations à tester
configs = [
    {'learning_rate': 0.001, 'optimizer': 'adam', 'layers': [64, 32]},
    {'learning_rate': 0.01, 'optimizer': 'adam', 'layers': [64, 32]},
    {'learning_rate': 0.0001, 'optimizer': 'adam', 'layers': [64, 32]},
    {'learning_rate': 0.001, 'optimizer': 'sgd', 'layers': [64, 32]},
    {'learning_rate': 0.001, 'optimizer': 'rmsprop', 'layers': [64, 32]},
    {'learning_rate': 0.001, 'optimizer': 'adam', 'layers': [128, 64, 32]},
    {'learning_rate': 0.001, 'optimizer': 'adam', 'layers': [32]},
]

# Stocker les résultats
results = []

for config in configs:
    print(f"Training with config: {config}")
    history, accuracy = train_and_evaluate(config['learning_rate'], config['optimizer'], config['layers'])
    results.append({'config': config, 'accuracy': accuracy})

# Afficher les résultats
for result in results:
    print(f"Config: {result['config']}, Accuracy: {result['accuracy']}")

# Tracer les courbes de précision pour la meilleure configuration
best_config = max(results, key=lambda x: x['accuracy'])
history, _ = train_and_evaluate(best_config['config']['learning_rate'], best_config['config']['optimizer'], best_config['config']['layers'])

sns.set_style("darkgrid")
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy en fonction de l\'epoch avec word embedding')
plt.legend()
plt.show()