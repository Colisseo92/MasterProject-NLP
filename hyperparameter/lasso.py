from sklearn.linear_model import Lasso
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../datas/lemmatized2.csv",nrows=10)

X = df["lemmatized"]
tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(X)

y = df["Score"]
y = y.replace(2,1)
y = y.replace(3,1)
y = y.replace(4,2)
y = y.replace(5,2)

X = tfidf_vectorizer.transform(X)

print("vectorized...")

names = tfidf_vectorizer.get_feature_names_out()

lasso = Lasso(alpha=0.1)
lasso_coef = lasso.fit(X, y).coef_
_ = plt.plot(range(len(names)), lasso_coef)
_ = plt.xticks(range(len(names)), names,
rotation=60)
_ = plt.ylabel('Coefficients')
plt.ylim(-0.0001,0.0001)
plt.show()