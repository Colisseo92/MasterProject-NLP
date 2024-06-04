import streamlit as st 
from joblib import dump,load
import numpy as np
import pandas as pd
import re
import string
import tensorflow as tf
from keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, GlobalAveragePooling1D
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from pywsd.utils import lemmatize_sentence

STARS = {1:"★☆☆☆☆",2:"★★☆☆☆",3:"★★★☆☆",4:"★★★★☆",5:"★★★★★"}

def transform_text(text: str):
    pattern = r'\b(?:{})\b'.format('|'.join(st.session_state["stopwords"]))
    url_pattern = r'https?://\S+|www\.\S'
    html_pattern =r'<[^>]*>'
    
    out = text.lower()
    out2 = re.sub(url_pattern, '', out)
    out3 = re.sub(html_pattern,'', out2)
    out4 = out3.translate(str.maketrans('', '', st.session_state["punctuation"]))
    out5 = re.sub(pattern,'',out4)
    out6 = re.sub(r'\d','',out5)
    out7 = re.sub(r'\s+',' ',out6)
    out8 = re.sub(r'\s+',' ',out7)
    return ' '.join(lemmatize_sentence(out8))

if 'stopwords' not in st.session_state:
    st.session_state["stopwords"] = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", 'amazon', 'product', 'products', 'br']

if 'punctuation' not in st.session_state:
    st.session_state["punctuation"] = string.punctuation
    
if 'vectorizer' not in st.session_state:
    tfidf_vectorizer = load("tfidf_vectorizer.joblib")
    st.session_state['vectorizer'] = tfidf_vectorizer
    
if 'LinearSVC' not in st.session_state:
    svc = load("svc.joblib")
    st.session_state['LinearSVC'] = svc
    
if 'NNvectorizer' not in st.session_state:
    df = pd.read_csv("lemmatized3.csv",nrows=10000)
    df.dropna(subset=["lemmatized", "Score"], inplace=True)

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
    st.session_state["NNvectorizer"] = vectorizer
    
if 'NeuralNetwork' not in st.session_state:
    df = pd.read_csv("lemmatized3.csv",nrows=10000)
    df.dropna(subset=["lemmatized", "Score"], inplace=True)

    X = df["lemmatized"]
    y = df["Score"]
    y = y.replace(1,0)
    y = y.replace(2,0)
    y = y.replace(3,0)
    y = y.replace(4,1)
    y = y.replace(5,1)
    
    max_words = 1000
    max_len = 100

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y,random_state=42)

    x_train = st.session_state['NNvectorizer'](X_train)
    x_test = st.session_state['NNvectorizer'](X_test)
    
    model = Sequential()
    model.add(Embedding(input_dim=max_words,output_dim=128,input_length=max_len))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = Adam(learning_rate=0.001)

    model.compile(loss="mse",metrics=['accuracy'],optimizer=optimizer)
    model.fit(x_train,y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))
    st.session_state['NeuralNetwork'] = model

st.title("Analyseur de Commentaire")
single_comment_tab, file_comment_tab = st.tabs(["Commentaire Unique","Fichier"])

with single_comment_tab:
    model = st.selectbox(
        "Quel model souhaitez-vous utiliser ?",
        ("LinearSVC","LogisticRegression","NeuralNetwork"),
        key = "tab1"
    )

    mode = st.selectbox(
        "Comment souhaitez-vous classifier les commentaires",
        ("Binaire (Bon/Mauvais)","Note (1-5)"),
        key = "tab1_1"
    )

    txt = st.text_area(
        "Commentaire à analyser",
        placeholder="Texte..."
    )

    if st.button("Vérifier le commentaire", type="primary"):
        predict = None
        if model in st.session_state:
            with st.spinner('Recherche de la réponse'):
                if model == "NeuralNetwork":
                    tfidf_text = st.session_state['NNvectorizer'](transform_text(txt))
                    print(tfidf_text)
                else:
                    tfidf_text = st.session_state['vectorizer'].transform([transform_text(txt)])
                if mode == "Binaire (Bon/Mauvais)":
                    predict = st.session_state[model].predict(tfidf_text)
                    print(predict)
                else:
                    if model == "LogisticRegression":
                        predict = st.session_state["LogisticRegressionMulti"].predict(tfidf_text)
            if mode == "Binaire (Bon/Mauvais)":
                if predict[0] == 2:
                    st.success('Commentaire positif')
                else:
                    st.error('Commentaire négatif')
            else:
                st.markdown("""
                <style>
                .big-font {
                    font-size:50px !important;
                }
                </style>
                """, unsafe_allow_html=True)

                st.markdown(f'<p class="big-font">{STARS[predict[0]]}</p>', unsafe_allow_html=True)
        else:
            st.warning("Le modèle selectionner semble ne pas être implémenté pour le moment !")
    
with file_comment_tab:
    file = st.file_uploader("Choose a file")
    if file is not None:
        dataframe = pd.read_csv(file)
        with st.spinner('Waiting ...'):
            tempo_text = dataframe["Text"].copy()
            dataframe["Transformer"] = tempo_text.apply(lambda x : transform_text(x))
            X = dataframe["Transformer"]
            tfidf_text = st.session_state['vectorizer'].transform(X)
            predict = st.session_state["LinearSVC"].predict(tfidf_text)
            predict_readable = []
            for i in predict:
                if i == 2:
                    predict_readable.append("Positif")
                else:
                    predict_readable.append("Négatif")
            dataframe["Prediction"] = predict_readable
            nombre_positif = predict_readable.count("Positif")
            nombre_negatif = predict_readable.count("Négatif")
            chart_data = pd.DataFrame(
                {
                    "Positif":[nombre_positif,0],
                    "Négatif":[0,nombre_negatif]
                }
            )
            st.dataframe(dataframe)
            st.bar_chart(chart_data)
            