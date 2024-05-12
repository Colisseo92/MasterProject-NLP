import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.svm import SVC
import numpy as np
from utils.plot import plot3dAccuracyPlot
import matplotlib as mpl
import matplotlib.pyplot as plt

RANDOM_STATE = 42

models = {
    "LogisticRegression": LogisticRegression(class_weight="balanced",random_state=RANDOM_STATE),
    "SVM": SVC(class_weight="balanced",random_state=RANDOM_STATE),
    "KNeighborsClassifier": KNeighborsClassifier(),
    "MLPClassifier": MLPClassifier(random_state=RANDOM_STATE),
    "RandomForestClassifier": RandomForestClassifier(class_weight="balanced",random_state=RANDOM_STATE),
    "DecisionTreeClassifier": DecisionTreeClassifier(class_weight="balanced",random_state=RANDOM_STATE),
    "Naives Bayes / MultinomialNB": MultinomialNB(),
    "Naives Bayes / BernoulliNB": BernoulliNB()
}

models_dict = {0:"LogisticRegression",1:"SVM",2:"KNeighborsClassifier",3:"MLPClassifier",4:"RandomForestClassifier",5:"DecisionTreeClassifier",6:"Naives Bayes / MultinomialNB",7:"Naives Bayes / BernoulliNB"}

def get_model_accuracy(model, X, y):
    cross_score = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    return cross_score


if __name__ == "__main__":
    df = pd.read_csv("datas/lemmatized.csv", nrows=10000)

    accuracy = []
    split = []
    model_list = []

    X = df['lemmatized']
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit(X)

    X = tfidf_vectorizer.transform(X)
    for i in range(2, 6):
        y = df['Score']
        if i == 2:
            y = y.replace(2, 1)
            y = y.replace(3, 1)
            y = y.replace(4, 2)
            y = y.replace(5, 2)
        if i == 3:
            y = y.replace(2, 1)
            y = y.replace(3, 2)
            y = y.replace(4, 3)
            y = y.replace(5, 3)
        if i == 4:
            y = y.replace(2, 1)

        model_number = 0
        for name, model in models.items():
            score = get_model_accuracy(model, X, y)
            print(f'{name} Score : {np.mean(score)}, StandardDeviation : {np.std(score)}')
            model_list.append(model_number)
            split.append(i)
            accuracy.append(np.mean(score))

            model_number += 1

    plot3dAccuracyPlot(accuracy,model_list,split,"test",list(models_dict.keys()),models_dict.values())