import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
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

end_time = time.time()
print("time elapsed: ", end_time-start_time)