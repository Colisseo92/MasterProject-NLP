import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import numpy as np

df = pd.read_csv("../datas/lemmatized.csv",nrows=10000)

X = df["lemmatized"]
y = df["Score"]
y = y.replace(2,1)
y = y.replace(3,1)
y = y.replace(4,2)
y = y.replace(5,2)

print("loaded...")

tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(X)

X = tfidf_vectorizer.transform(X)

print("transformed....")

params = {
    'penalty':[None],
    'class_weight':['balanced',None],
    'solver':['saga','sag','lbfgs','newton-cholesky','newton-cg']
}

logreg = LogisticRegression(random_state=42,max_iter=100)

search = GridSearchCV(logreg,params,cv=5,scoring="roc_auc")
search.fit(X,y)

#{'C': 1, 'class_weight': None, 'penalty': 'l2', 'solver': 'saga'}
#{'class_weight': None, 'penalty': None, 'solver': 'lbfgs'}
#{'class_weight': None, 'penalty': None, 'solver': 'saga'}
#{'class_weight': 'balanced', 'penalty': 'l1', 'solver': 'liblinear'}

print(search.best_params_)