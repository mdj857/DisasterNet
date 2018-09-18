'''
This file performs Latent Dirichlet Analysis on image captions
from images from Hurricane Harvey
'''

from __future__ import unicode_literals
import numpy as np
import pandas as pd
import re, nltk, gensim
import spacy

# Sklearn
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint

# Plotting tools
import pyLDAvis
import pyLDAvis.sklearn
import matplotlib.pyplot as plt

'''
This function takes a list of documents and tokenizes each document, 
removing the punctuation in the process.
'''
def doc_to_words(documents):
    for document in documents:
        yield(gensim.utils.simple_preprocess(str(document), deacc=True))  # deacc=True removes punctuations

'''
This function lemmatizes each token in the list of lists using the spacy.io 
package. (https://spacy.io/api/annotation)
Note that the strings were converted to Unicode to preserve compatibiliy with 
spacy.io
'''
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]))
    return texts_out


'''SCRIPT START'''
df = pd.read_csv("google_cloud_image_annotations.csv")

#convert corpora to list
data = df.iloc[:,1].values.tolist()

# remove quotes that were used to allow documents to be single elements in the .csv
data = [re.sub('"', '', tags) for tags in data]

# convert each text to unicode
data = [doc.decode('utf-8') for doc in data]

# clean the corpora and tokenize 
data_words = list(doc_to_words(data))

# because the google vision API could spit out tags containing the same tokens 
# (i.e 'motor vehicles' and 'vehicles'), i'll remove the duplicates (i.e only keep the )

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# Run in terminal: python3 -m spacy download en
nlp = spacy.load('en', disable=['parser', 'ner'])

# Do lemmatization keeping only Noun, Adj, Verb, Adverb
data_lemmatized = lemmatization(data_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])


