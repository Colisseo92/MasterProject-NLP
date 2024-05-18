import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import numpy as np

df = pd.read_csv("../datas/lemmatized.csv")

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
    'penalty':['l1','l2'],
    'class_weight':['balanced',None],
    'loss':['hinge','squared_hinge'],
    'C':[0.1,1,10,100]
}

#{'C': 1, 'class_weight': 'balanced', 'loss': 'squared_hinge', 'penalty': 'l2'}

svc = LinearSVC(random_state=42,max_iter=3000)

search = GridSearchCV(svc,params,cv=5,scoring="roc_auc",error_score=0.0)
search.fit(X,y)

print(search.best_params_)