#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional


# In[2]:


data = pd.read_csv('C:\\Users\\suzai\\reviews_26.csv', encoding='latin1')


# In[3]:


stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if isinstance(text, str):
        # Tokenize the text
        words = word_tokenize(text)
        # Convert to lower case
        words = [word.lower() for word in words]
        # Remove stopwords
        words = [word for word in words if word.isalnum() and word not in stop_words]
        return ' '.join(words)
    return ''

data['cleaned_review'] = data['Review'].apply(preprocess_text)


# In[4]:


print(data['cleaned_review'])


# In[5]:


labels = data['sentiment'].values
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(labels)
categorical_labels = to_categorical(encoded_labels, num_classes=3)


# In[6]:


tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(data['cleaned_review'].values)
sequences = tokenizer.texts_to_sequences(data['cleaned_review'].values)
padded_sequences = pad_sequences(sequences, maxlen=200)


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(padded_sequences, categorical_labels, test_size=0.2, random_state=2)


# In[8]:


model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=128))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(32))
model.add(Dense(3, activation='softmax'))


# In[9]:


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[10]:


model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2)


# In[12]:


loss, accuracy = model.evaluate(X_test, y_test)


# In[ ]:




