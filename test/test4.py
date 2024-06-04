# -*- coding: utf-8 -*-
"""
Created on Fri May  3 17:47:42 2024

@author: Flavi
"""

import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.model_selection import TimeSeriesSplit
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("lemmatized2.csv")
df2 = df[df.Score == 1]


X = df["lemmatized"]
y = df["Score"]
y = y.replace(2,1)
y = y.replace(3,1)
y = y.replace(4,2)
y = y.replace(5,2)

vocab = ['addict' 'albeit' 'alright' 'always' 'amaze' 'amazing' 'annoy' 'assume'
 'away' 'awful' 'bad' 'barely' 'beautiful' 'beautifully' 'become'
 'benefit' 'best' 'bisquit' 'blah' 'bland' 'booty' 'bottom' 'brother'
 'button' 'caffein' 'calm' 'cancel' 'carbonated' 'care' 'careful' 'carmel'
 'carry' 'cause' 'certain' 'character' 'cheddar' 'chicory' 'cigar' 'claim'
 'clearly' 'concept' 'convert' 'crayon' 'crisp' 'crispy' 'crunchies'
 'culture' 'date' 'de' 'dead' 'decent' 'deceptive' 'deliberately'
 'delicious' 'delight' 'dented' 'description' 'design' 'didnt' 'die'
 'digestive' 'disappears' 'disappoint' 'disappointed' 'disappointing'
 'disappointment' 'divine' 'donate' 'door' 'dop' 'downside' 'drinkbr'
 'dump' 'duncan' 'earths' 'edible' 'essentially' 'even' 'excellent'
 'exceptionally' 'excited' 'fabulous' 'fantastic' 'fast' 'faves'
 'favorite' 'favorites' 'fee' 'feel' 'felt' 'flat' 'flavorless' 'folk'
 'fourth' 'freind' 'fun' 'gag' 'gallon' 'garbage' 'gfdf' 'giant' 'glad'
 'good' 'google' 'great' 'gritty' 'ground' 'guess' 'habit' 'halloween'
 'halo' 'happy' 'hasnt' 'havebr' 'hershey' 'hesitant' 'highly' 'hines'
 'hook' 'hooked' 'hop' 'horrible' 'hubby' 'hull' 'hurt' 'hype' 'idea'
 'identify' 'impact' 'import' 'inquiry' 'inside' 'kettle' 'lack' 'leaks'
 'letter' 'lie' 'likely' 'likes' 'lousy' 'love' 'luck' 'marginally'
 'maybe' 'mediocre' 'mesh' 'microwave' 'mislead' 'misleading' 'mistake'
 'mixer' 'moist' 'moldy' 'money' 'moneybr' 'morning' 'mountain' 'mouthful'
 'msg' 'mth' 'mushy' 'mustard' 'nearly' 'neither' 'nestle' 'nice' 'nog'
 'nogo' 'notbr' 'notify' 'offensive' 'office' 'ok' 'okay' 'okbut' 'omega'
 'onion' 'opportunity' 'overcook' 'overdose' 'overprice' 'packbr'
 'partially' 'penny' 'percentage' 'perfect' 'perfectly' 'picture' 'pitch'
 'pith' 'pleasantly' 'pleased' 'plockys' 'pregnant' 'preserve' 'pricing'
 'print' 'programbr' 'properly' 'quickly' 'rancid' 'rather' 'realize'
 'receive' 'recipe' 'recommend' 'refresh' 'refund' 'relate' 'relation'
 'return' 'ripoff' 'rjs' 'roof' 'ruin' 'run' 'salon' 'sand' 'satisfy'
 'save' 'seth' 'shame' 'shiny' 'sickeningly' 'sits' 'skeptical' 'smallist'
 'smokies' 'smooth' 'snack' 'sorry' 'soso' 'sothere' 'specialty' 'stale'
 'starbucks' 'steve' 'stick' 'stir' 'strongbr' 'sumatra' 'sunny'
 'supplement' 'supply' 'swiss' 'tang' 'tangythis' 'tap' 'taste'
 'tasteless' 'taught' 'term' 'terrible' 'terribly' 'thank' 'thanks'
 'think' 'thought' 'throw' 'tortilla' 'transaction' 'trash' 'tullys' 'ugh'
 'undone' 'unfortunately' 'unless' 'unlike' 'untested' 'uprise' 'variance'
 'versus' 'vet' 'vinegary' 'vitamin' 'vomit' 'wake' 'walmart' 'washington'
 'waste' 'watered' 'watermelonstrawberry' 'watery' 'weak' 'wheres'
 'without' 'wonderful' 'wonderfully' 'wont' 'workout' 'worry' 'worst'
 'worthless' 'would' 'xanthan' 'year' 'yuck' 'yucky' 'yummy' 'zero']
#########TEST PERFORMANCES##########

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

tfidf_vectorizer = TfidfVectorizer(vocabulary=vocab)
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
plt.title("Confusion Matrix for LinearSVC")
plt.savefig("test2.png")