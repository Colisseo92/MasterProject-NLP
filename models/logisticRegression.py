import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("../datas/lemmatized2.csv")

X = df["lemmatized"]
y = df["Score"]
y = y.replace(2,1)
y = y.replace(3,1)
y = y.replace(4,2)
y = y.replace(5,2)

print("loaded...")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(X_train)

X_train_tfidf = tfidf_vectorizer.transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

##{'C': 1, 'class_weight': None, 'penalty': 'l2', 'solver': 'saga'} v
#{'class_weight': None, 'penalty': None, 'solver': 'lbfgs'}
#{'class_weight': None, 'penalty': None, 'solver': 'saga'}
#{'class_weight': 'balanced', 'penalty': 'l1', 'solver': 'liblinear'}
logreg = LogisticRegression(solver='lbfgs',penalty=None,class_weight=None,random_state=42,max_iter=3000)
print("created...")
logreg.fit(X_train_tfidf, y_train)

y_pred = logreg.predict(X_test_tfidf)

print(classification_report(y_test,y_pred,digits=3))
ConfusionMatrixDisplay.from_predictions(y_test,y_pred,cmap=plt.cm.Blues)
plt.title("Confusion Matrix for Logistic Regression avec optimisation des paramètres")
plt.show()