from __future__ import unicode_literals, division
import numpy as np
import pandas as pd
import re, nltk, gensim
import spacy
import clean_corpus as clean

'''SCRIPT START'''
df = pd.read_csv("google_cloud_image_annotations.csv")

#convert corpora to list
data = df.iloc[:,1].values.tolist()

# remove quotes that were used to allow documents to be single elements in the .csv
data = [re.sub('"', '', tags) for tags in data]

# convert each text to unicode
data = [doc.decode('utf-8') for doc in data]

# clean the corpora and tokenize 
data_words = list(clean.doc_to_words(data))

# because the google vision API could spit out tags containing the same tokens 
# (i.e 'motor vehicles' and 'vehicles'), I could remove the duplicates (i.e only keep the )

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
nlp = spacy.load('en', disable=['parser', 'ner'])

# Do lemmatization keeping only Noun, Adj, Verb, Adverb
data_lemmatized = clean.lemmatization(data_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])