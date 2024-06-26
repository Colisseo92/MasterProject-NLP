import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("../datas/lemmatized3.csv")
df.dropna(subset=["lemmatized", "Score"], inplace=True)

X = df["lemmatized"]
y = df["Score"]
y = y.replace(1,0)
y = y.replace(2,0)
y = y.replace(3,0)
y = y.replace(4,1)
y = y.replace(5,1)

print("loaded...")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(X_train)

X_train_tfidf = tfidf_vectorizer.transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

#{'C': 0.1, 'class_weight': 'balanced', 'loss': 'squared_hinge', 'penalty': 'l2'}
svc = LinearSVC(penalty='l2',loss="squared_hinge",C=1,class_weight=None,random_state=42)
print("created...")

svc.fit(X_train_tfidf, y_train)

y_pred = svc.predict(X_test_tfidf)

print(classification_report(y_test,y_pred,digits=3))
ConfusionMatrixDisplay.from_predictions(y_test,y_pred,cmap=plt.cm.Blues)
plt.title("penalty: 'l2', loss: 'squared_hinge', C: 1, class_weight: None")
plt.show()