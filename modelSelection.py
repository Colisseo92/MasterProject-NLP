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
score_dict = {0:"accuracy",1:"precision",2:"f1",3:"recall",4:"roc_auc"}

def get_model_accuracy(model, X, y,score):
    cross_score = cross_val_score(model, X, y, cv=5, scoring=score)
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
    y = df['Score']
    y = y.replace(2, 1)
    y = y.replace(3, 1)
    y = y.replace(4, 2)
    y = y.replace(5, 2)

    model_number = 0
    for name, model in models.items():
        for number, score in score_dict.items():
            result = get_model_accuracy(model, X, y, score)
            print(f'{name} {score} : {np.mean(result)}, StandardDeviation : {np.std(result)}')
            model_list.append(model_number)
            split.append(number)
            accuracy.append(np.mean(result))

        model_number += 1

    plot3dAccuracyPlot(accuracy,model_list,split,"test",list(models_dict.keys()),models_dict.values(),"precision")