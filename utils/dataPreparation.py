import time
import pandas as pd
import string
import nltk
import re
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from pywsd.utils import lemmatize_sentence

def getWordCloud(dataframe, column_name: str, word_number: int):
    all_text = " ".join(dataframe[column_name].dropna())
    words = all_text.split(" ")
    words_count = pd.Series(words).value_counts()

    word_freq = {}
    for element in zip(words_count.head(word_number).index.tolist(), words_count.head(word_number).values.tolist()):
        word_freq[element[0]] = element[1]

    wordcloud = WordCloud(width=600, height=300, background_color="white").generate_from_frequencies(word_freq)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.show()

start_time = time.time()

PUNCTUATION = string.punctuation

def getStopWordRegex():
    stopwords_default = stopwords.words('english')
    stopwords_default.append("amazon")
    stopwords_default.append("product")
    stopwords_default.append("products")
    stopwords_default.append("br")

    pattern = r'\b(?:{})\b'.format('|'.join(stopwords_default))
    return pattern

def removeStopWords(text):
    stop_word_pattern = re.compile(getStopWordRegex())
    return stop_word_pattern.sub(r'', text)

def removeUrls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def removePunctuation(text):
    return text.translate(str.maketrans('', '', PUNCTUATION))

def removeNumbers(text):
    number_pattern = re.compile(r'\d')
    return number_pattern.sub(r'', text)

def removeSpaces(text):
    space_pattern = re.compile(r'\s+')
    return space_pattern.sub(r' ', text)

df = pd.read_csv('../datas/Reviews.csv')

df["Text"] = df["Text"].str.lower()
df["Text"] = df["Text"].apply(lambda text: removeUrls(text))
df["Text"] = df["Text"].apply(lambda text: removeStopWords(text))
df["Text"] = df["Text"].apply(lambda text: removePunctuation(text))
df["Text"] = df["Text"].apply(lambda text: removeNumbers(text))
df["Text"] = df["Text"].apply(lambda text: removeSpaces(text))

df["lemmatized"] = pd.Series(df["Text"].values).map(lambda x: ' '.join(lemmatize_sentence(x)))
df.to_csv('lemmatized3.csv')
end_time = time.time()

print(f"Execution Time : {end_time - start_time}")