from sklearn.feature_selection import RFE
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

df = pd.read_csv("../datas/lemmatized2.csv")

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
print("transformed")

logreg = LogisticRegression(solver='lbfgs',penalty=None,class_weight=None,random_state=42,max_iter=3000)
print("created...")
rfe_selector = RFE(estimator=logreg,n_features_to_select=5)

X_new = rfe_selector.fit_transform(X,y)

selected_features = rfe_selector.support_
print(selected_features)
