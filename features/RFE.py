from sklearn.feature_selection import RFE
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
import numpy as np

df = pd.read_csv("../datas/lemmatized3.csv",nrows=10000)
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
X_new = tfidf_vectorizer.transform(X)
print("transformed")

logreg = svc = LinearSVC(penalty='l2',loss="squared_hinge",C=1,class_weight=None,random_state=42)
print("created...")
rfe_selector = RFE(estimator=logreg,n_features_to_select=300)

a = rfe_selector.fit_transform(X_new,y)
selected_features = rfe_selector.get_support(indices=True)
best_feature_vocab = np.array(tfidf_vectorizer.get_feature_names_out())[selected_features]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

vectorizer = TfidfVectorizer(vocabulary=best_feature_vocab)
vectorizer.fit(X_train)

X_train_tfidf = vectorizer.transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

svc = LinearSVC(penalty='l2',loss="squared_hinge",C=1,class_weight=None,random_state=42)
print("created...")

svc.fit(X_train_tfidf, y_train)

y_pred = svc.predict(X_test_tfidf)

print(classification_report(y_test,y_pred,digits=3))
ConfusionMatrixDisplay.from_predictions(y_test,y_pred,cmap=plt.cm.Blues)
plt.title("Confusion Matrix LinearSVC avec SelectKBest - f_classif k = 300")
plt.show()

print(best_feature_vocab)
