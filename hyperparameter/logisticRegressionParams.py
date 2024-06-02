import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import numpy as np
start_time = time.time()

df = pd.read_csv("../datas/lemmatized3.csv",nrows=100000)
df.dropna(subset=["lemmatized", "Score"], inplace=True)

X = df["lemmatized"]
y = df["Score"]
y = y.replace(1,0)
y = y.replace(2,0)
y = y.replace(3,0)
y = y.replace(4,1)
y = y.replace(5,1)

tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(X)

X = tfidf_vectorizer.transform(X)

print("transformed....")

params = {
    'penalty':['l2'],
    'class_weight':[None,"balanced"],
    'solver':['lbfgs','liblinear','newton-cg','newton-cholesky','sag','saga']
}

params2 = {
    'penalty':['l1'],
    'class_weight':[None,"balanced"],
    'solver':['liblinear','saga']
}
params3 = {
    'penalty':[None],
    'class_weight':[None,"balanced"],
    'solver':['lbfgs','newton-cg','newton-cholesky','sag','saga']
}

logreg = LogisticRegression(random_state=42,max_iter=3000)

search = GridSearchCV(logreg,params,cv=5,scoring="roc_auc",error_score=0)
search.fit(X,y)

#{'C': 1, 'class_weight': None, 'penalty': 'l2', 'solver': 'saga'}
#{'class_weight': None, 'penalty': None, 'solver': 'lbfgs'}
#{'class_weight': None, 'penalty': None, 'solver': 'saga'}
#{'class_weight': 'balanced', 'penalty': 'l1', 'solver': 'liblinear'}

print(search.best_params_)
end_time = time.time()
print(end_time-start_time)