#X features = a text
#Y label = a category
#prepare data e.g., tf-idf, bag of words, remove stop words, lemmatization, etc.
#train-test split
#model training
#data testing

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

df = pd.read_csv("spam_data.csv")
df.head()

#check categories
df["Category"].value_counts() #unbalanced dataset, use Naive Bayes ComplementNB

X = df["Message"]
y = df["Category"]

#train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
len(X_train), len(X_test)  # Check the number of samples in train and test sets

#create pipeline for multinomial naive bayes, complement naive bayes, and support vector classifier

pipeMNB = Pipeline([("tfidf", TfidfVectorizer()),("clf", MultinomialNB())])
#pipeMNB = Pipeline([("tfidf", TfidfVectorizer(stop_words="english")),("clf", MultinomialNB())])

pipeCNB = Pipeline([("tfidf", TfidfVectorizer()),("clf", ComplementNB())])
#pipeSVC = Pipeline([("tfidf", TfidfVectorizer()),("clf", SVC())])
pipeSVC = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 3))),("clf", SVC())])

#fit MNB 
pipeMNB.fit(X_train, y_train)

#test MNB
predMNB = pipeMNB.predict(X_test)
predMNB
print(f"MNB:{accuracy_score(y_test, predMNB):.2f}")#97%
print(classification_report(y_test, predMNB))  # Detailed classification report
print(confusion_matrix(y_test, predMNB))  # Confusion matrix

#fit ComplementNB
pipeCNB.fit(X_train, y_train)       

#test ComplementNB
predCNB = pipeCNB.predict(X_test)
print(f"ComplementNB:{accuracy_score(y_test, predCNB):.2f}")
print(classification_report(y_test, predCNB))  # Detailed classification report
print(confusion_matrix(y_test, predCNB))  # Confusion matrix

#fit SVC
pipeSVC.fit(X_train, y_train)       

#test SVC
predSVC = pipeSVC.predict(X_test)
print(f"SVC:{accuracy_score(y_test, predSVC):.2f}")
print(classification_report(y_test, predSVC))  # Detailed classification report
print(confusion_matrix(y_test, predSVC))  # Confusion matrix    


#test with a message
message = ["Congratulations! You've won a lottery! Claim your prize now."]
pred_message = pipeSVC.predict(message)
print(pred_message)

