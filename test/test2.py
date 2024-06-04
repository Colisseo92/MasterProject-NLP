import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("lemmatized2.csv",nrows=10000)

X=df["lemmatized"]
y = df["Score"]
y= y.replace(2,1)
y = y.replace(3,1)
y = y.replace(4,2)
y = y.replace(5,2)

print("loaded")

tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(X)

X = tfidf_vectorizer.transform(X)

svc = LinearSVC(penalty="l2",loss="squared_hinge",C=1,class_weight=None,random_state=42)
print("created")
rfe_selector = RFE(estimator=svc, n_features_to_select=50)
print("created")
# Fit and transform the data
X_new = rfe_selector.fit_transform(X, y)
print("transformed")

# Selected features
selected_features = rfe_selector.support_
voc_id = [i for i,feature in enumerate(selected_features) if feature == True]
voc = np.array(tfidf_vectorizer.get_feature_names_out())[voc_id]

X_train, X_test, y_train, y_test = train_test_split(df["lemmatized"],y,test_size=0.2,stratify=y,random_state=42)

vectorizer = TfidfVectorizer(vocabulary=voc)
vectorizer.fit(X_train)

X_train_tfidf = vectorizer.transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

svc2 = svc = LinearSVC(penalty="l2",loss="squared_hinge",C=1,class_weight=None,random_state=42)
svc2.fit(X_train_tfidf,y_train)

y_pred = svc.predict(X_test_tfidf)

print(classification_report(y_test,y_pred,digits=3))
ConfusionMatrixDisplay.from_predictions(y_test,y_pred,cmap=plt.cm.Blues)
plt.title("Confusion Matrix for LinearSVC")
plt.savefig("plot.png")

vectorizer = TfidfVectorizer()
vectorizer.fit(X_train)

X_train_tfidf = vectorizer.transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

svc2 = svc = LinearSVC(penalty="l2",loss="squared_hinge",C=1,class_weight=None,random_state=42)
svc2.fit(X_train_tfidf,y_train)

y_pred = svc.predict(X_test_tfidf)

print(classification_report(y_test,y_pred,digits=3))