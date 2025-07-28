# General packages and dictionary analysis
import os
import tarfile
import bz2
import urllib.request
import re
import pickle
import nltk
#nltk.download("stopwords")
import eli5
import joblib
import requests
import pandas as pd
import numpy as np
from numpy import array
from numpy import asarray
from numpy import zeros
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import spacy
from textacy import preprocessing
from functools import partial

import tensorflow
from tensorflow.keras.preprocessing.text import one_hot, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Dense
from keras.layers import Flatten, GlobalMaxPooling1D, Embedding, Conv1D, LSTM
from sklearn.model_selection import train_test_split

filename = "reviewdata.pickle.bz2"
if os.path.exists(filename):
    print(f"Using cached file {filename}")
    with bz2.BZ2File(filename, "r") as zipfile:
        data = pickle.load(zipfile)
    text_train, text_test, y_train, y_test = data
else:
    url = "https://cssbook.net/d/aclImdb_v1.tar.gz"
    print(f"Downloading from {url}")
    fn, _headers = urllib.request.urlretrieve(url, filename=None)
    t = tarfile.open(fn, mode="r:gz")
    text_train, text_test = [], []
    y_train, y_test = [], []
    for f in t.getmembers():
        m = re.match("aclImdb/(\w+)/(pos|neg)/", f.name)
        if not m:
            # skip folder names, other categories
            continue
        dataset, label = m.groups()
        text = t.extractfile(f).read().decode("utf-8")
        if dataset == "train":
            text_train.append(text)
            y_train.append(label)
        elif dataset == "test":
            text_test.append(text)
            y_test.append(label)
    data = text_train, text_test, y_train, y_test
    print(f"Saving to {filename}")
    with bz2.BZ2File(filename, "w") as zipfile:
        pickle.dump(data, zipfile)

# I want to have only two columns: text and label, mixing train and test sets
df = pd.DataFrame({
    "review": text_train + text_test,
    "sentiment": y_train + y_test
})
df["sentiment"] = df["sentiment"].map({"pos": 1, "neg": 0})
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle the dataset
print(f"Dataset size: {len(df)}")   


#check null values 
df.isnull().sum()

#check distribution of sentiment
df["sentiment"].value_counts().plot(kind="bar", title="Sentiment distribution")

# Remove HTML tags
tag_re = re.compile(r"<[^>]+>")
def remove_tags(text):
    return tag_re.sub("", text)


debug = False

def preprocess_text(sen):
    '''Cleans up text data '''

    sentence = sen.lower()  # Lowercase
    
    sentence = remove_tags(sentence)  # Remove HTML tags

    if debug:
        print(f"After removing tags: {sentence}")
    sentence = re.sub("[^a-zA-Z\s]", "", sentence)  # Remove non-alphabetic characters
    if debug:
        print(f"After removing non-alphabetic characters: {sentence}")
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)  # When we remove apostrophe from the word "Mark's".
    if debug:
        print(f"After removing single characters: {sentence}")
    sentence = re.sub(r"\s+", ' ', sentence)  # Remove extra spaces
    if debug:
        print(f"After removing extra spaces: {sentence}")
    pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')

    if debug:
        print(f"Stopwords pattern: {pattern}")
    sentence = pattern.sub('', sentence)
    return sentence

X = []
sentences = list(df['review'])
#sentences = [df['review'][0], df["review"][1]]  # Note: 'sencences' is a typo, should be 'sentences'
for sen in sentences:
    X.append(preprocess_text(sen))

# Display the first few preprocessed sentences
print(X[:2])

# get labels
y = df['sentiment']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# The train set will be used to train our deep learning models 
# while test set will be used to evaluate how well our model performs 

# Embedding layer expects the words to be in numeric form 
# Using Tokenizer function from keras.preprocessing.text library
# Method fit_on_text trains the tokenizer 
# Method texts_to_sequences converts sentences to their numeric form

word_tokenizer = Tokenizer()
word_tokenizer.fit_on_texts(X_train)

X_train = word_tokenizer.texts_to_sequences(X_train)
X_test = word_tokenizer.texts_to_sequences(X_test)

# Adding 1 to store dimensions for words for which no pretrained word embeddings exist

vocab_length = len(word_tokenizer.word_index) + 1
vocab_length


# Padding all reviews to fixed length 100
# This is done to ensure that all reviews have the same length, which is required for training deep learning models.
maxlen = 100 #chosen arbitrarily, can be changed
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

# Load GloVe word embeddings and create an Embeddings Dictionary
# GloVe is a pre-trained word embedding model (on 6 billion tokens in this case) that provides vector representations of words.
#each word is represented by a 100-dimensional vector,  but you can choose other dimensions
# Download GloVe embeddings from https://nlp.stanford.edu/projects/glove/

embeddings_dictionary = dict() # Create an empty dictionary to hold the embeddings
glove_file = open('a2_glove.6B.100d.txt', encoding="utf8")

for line in glove_file: #goes through each line in the GloVe file
    records = line.split() #splits the line into words
    word = records[0] #first word is the word itself
    vector_dimensions = asarray(records[1:], dtype='float32') #remaining word is the vector dimensions
    embeddings_dictionary [word] = vector_dimensions
glove_file.close()

#example of how to access the embeddings for the word "good"
#embeddings_dictionary["good"]

# Create Embedding Matrix having 100 columns (for 100-dimensional GloVe embeddings)
# Containing word embeddings for all words in our corpus.
#the model will use this matrix to understand semantic meaning of words 

embedding_matrix = zeros((vocab_length, 100)) #start with a matrix of zeros
for word, index in word_tokenizer.word_index.items(): #goes through each word and its index
    embedding_vector = embeddings_dictionary.get(word) #gets the embedding vector for the word from the dictionary
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector #if the embedding vector is not None, assigns it to the matrix at the index of the word


# Neural Network architecture
# Simple neural network
snn_model = Sequential()
embedding_layer = Embedding(vocab_length, 100, weights=[embedding_matrix], input_length=maxlen , trainable=False)

snn_model.add(embedding_layer)

snn_model.add(Flatten())
snn_model.add(Dense(1, activation='sigmoid'))

# Model compiling
snn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

print(snn_model.summary())

# Model training
snn_model_history = snn_model.fit(X_train, y_train, batch_size=128, epochs=6, verbose=1, validation_split=0.2)

# Predictions on the Test Set
score = snn_model.evaluate(X_test, y_test, verbose=1)

# Model Performance
print("Test Score:", score[0])
print("Test Accuracy:", score[1])

# Model Performance Charts
import matplotlib.pyplot as plt

plt.plot(snn_model_history.history['acc'])
plt.plot(snn_model_history.history['val_acc'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

plt.plot(snn_model_history.history['loss'])
plt.plot(snn_model_history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

#Convolutional Neural Network

cnn_model = Sequential()

embedding_layer = Embedding(vocab_length, 100, weights=[embedding_matrix], input_length=maxlen , trainable=False)
cnn_model.add(embedding_layer)

cnn_model.add(Conv1D(128, 5, activation='relu'))
cnn_model.add(GlobalMaxPooling1D())
cnn_model.add(Dense(1, activation='sigmoid'))

# Model compiling
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
print(cnn_model.summary())

# Model training
cnn_model_history = cnn_model.fit(X_train, y_train, batch_size=128, epochs=6, verbose=1, validation_split=0.2)

# Predictions on the Test Set
score = cnn_model.evaluate(X_test, y_test, verbose=1)

# Model Performance
print("Test Score:", score[0])
print("Test Accuracy:", score[1])

# Model Performance Charts
plt.plot(cnn_model_history.history['acc'])
plt.plot(cnn_model_history.history['val_acc'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc = 'upper left')
plt.show()

plt.plot(cnn_model_history.history['loss'])
plt.plot(cnn_model_history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc = 'upper left')
plt.show()

#LSTM Model

lstm_model = Sequential()
embedding_layer = Embedding(vocab_length, 100, weights=[embedding_matrix], input_length=maxlen , trainable=False)

lstm_model.add(embedding_layer)
lstm_model.add(LSTM(128))

lstm_model.add(Dense(1, activation='sigmoid'))

# Model compiling
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
print(lstm_model.summary())

# Model Training
lstm_model_history = lstm_model.fit(X_train, y_train, batch_size=128, epochs=6, verbose=1, validation_split=0.2)


# Predictions on the Test Set
score = lstm_model.evaluate(X_test, y_test, verbose=1)

# Model Performance
print("Test Score:", score[0])
print("Test Accuracy:", score[1])

# Model Performance Charts
plt.plot(lstm_model_history.history['acc'])
plt.plot(lstm_model_history.history['val_acc'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

plt.plot(lstm_model_history.history['loss'])
plt.plot(lstm_model_history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

#Making predictions on new data
# Load sample IMDb reviews csv, having ~6 movie reviews, along with their IMDb rating

sample_reviews = pd.read_csv("unseen_reviews.csv")

sample_reviews.head(6)

# Preprocess review text with earlier defined preprocess_text function

unseen_reviews = sample_reviews['Review Text']

unseen_processed = []
for review in unseen_reviews:
  review = preprocess_text(review)
  unseen_processed.append(review)

# Tokenising instance with earlier trained tokeniser
unseen_tokenized = word_tokenizer.texts_to_sequences(unseen_processed)

# Pooling instance to have maxlength of 100 tokens
unseen_padded = pad_sequences(unseen_tokenized, padding='post', maxlen=maxlen)

# Passing tokenised instance to the LSTM model for predictions
unseen_sentiments = lstm_model.predict(unseen_padded)

unseen_sentiments

# Writing model output file back 

sample_reviews['Predicted Sentiments'] = np.round(unseen_sentiments*10,1)

df_prediction_sentiments = pd.DataFrame(sample_reviews['Predicted Sentiments'], columns = ['Predicted Sentiments'])
df_movie                 = pd.DataFrame(sample_reviews['Movie'], columns = ['Movie'])
df_review_text           = pd.DataFrame(sample_reviews['Review Text'], columns = ['Review Text'])
df_imdb_rating           = pd.DataFrame(sample_reviews['IMDb Rating'], columns = ['IMDb Rating'])


dfx=pd.concat([df_movie, df_review_text, df_imdb_rating, df_prediction_sentiments], axis=1)

dfx.to_csv("./unseen_predictions.csv", sep=',', encoding='UTF-8')

dfx.head(6)