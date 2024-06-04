import re
import string
import pandas as pd
import tensorflow as tf
from pywsd.utils import lemmatize_sentence

df = pd.read_csv('test.csv')
STOPWORDS = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", 'amazon', 'product', 'products', 'br']

def transform_text(text: str):
    pattern = r'\b(?:{})\b'.format('|'.join(STOPWORDS))
    url_pattern = r'https?://\S+|www\.\S'
    html_pattern =r'<[^>]*>'
    
    out = text.lower()
    out2 = re.sub(url_pattern, '', out)
    out3 = re.sub(html_pattern,'', out2)
    out4 = out3.translate(str.maketrans('', '', string.punctuation))
    out5 = re.sub(pattern,'',out4)
    out6 = re.sub(r'\d','',out5)
    out7 = re.sub(r'\s+',' ',out6)
    out8 = re.sub(r'\s+',' ',out7)
    return ' '.join(lemmatize_sentence(out8))

print(transform_text('My dog doesn\'t like this food. He barely touches it.'))

model = tf.keras.models.load_model('model.keras')