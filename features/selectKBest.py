import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, f_classif
import numpy as np
start_time = time.time()

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(X_train)

X_train_tfidf = tfidf_vectorizer.transform(X_train)

#SELECTION DES MEILLEURS FEATURES
best_feature = SelectKBest(chi2 , k=300).fit(X_train_tfidf, y_train).get_support(indices=True)
best_feature_vocab = np.array(tfidf_vectorizer.get_feature_names_out())[best_feature]

#####" AVEC FEATURE SELECT

vectorizer = TfidfVectorizer(vocabulary=best_feature_vocab)
vectorizer.fit(X_train)

X_train_tfidf_2 = vectorizer.transform(X_train)
X_test_tfidf_2 = vectorizer.transform(X_test)

svc = LinearSVC(penalty='l2',loss="squared_hinge",C=1,class_weight=None,random_state=42)
print("created...")

svc.fit(X_train_tfidf_2, y_train)

y_pred = svc.predict(X_test_tfidf_2)

print(classification_report(y_test,y_pred,digits=3))
ConfusionMatrixDisplay.from_predictions(y_test,y_pred,cmap=plt.cm.Blues)
plt.title("Confusion Matrix LinearSVC avec SelectKBest - f_classif k = 300")
plt.show()

end_time = time.time()
print(end_time-start_time)