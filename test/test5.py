from joblib import dump,load
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

df = pd.read_csv("lemmatized2.csv")

df2 = df[df.Score == 1]
X = df2["lemmatized"]
y = df2["Score"]

y_list = y.tolist()

tfidf_vectorizer = load("tfidf_vectorizer.joblib")
X_transform = tfidf_vectorizer.transform(X)

logreg = load("logreg.joblib")

y_predict_o = logreg.predict_proba(X_transform)

x = [proba[0] for proba in y_predict_o]
y_z = [proba[1] for proba in y_predict_o]

print(x)
print(y_z)

plt.plot(x,y_z)
plt.savefig("scatter.png")
