'''
This file performs Latent Dirichlet Analysis on image captions
from images from Hurricane Harvey
'''

from __future__ import unicode_literals, division
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
# (i.e 'motor vehicles' and 'vehicles'), I could remove the duplicates (i.e only keep the )

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
nlp = spacy.load('en', disable=['parser', 'ner'])

# Do lemmatization keeping only Noun, Adj, Verb, Adverb
data_lemmatized = lemmatization(data_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])


# create term-document matrix
vectorizer = CountVectorizer(analyzer='word',       
                             min_df=10,                        # minimum reqd occurences of a word 
                             stop_words='english',             # remove stop words
                             lowercase=True,                   # convert all words to lowercase
                             token_pattern='[a-zA-Z0-9]{3,}',  # num chars > 3
                             max_features=50000,               # max number of uniq words
                            )

data_vectorized = vectorizer.fit_transform(data_lemmatized)

# project data into 2-D
data_dense = data_vectorized.todense()
print data_dense

# Compute Sparsicity = Percentage of Non-Zero cells -- maybe this'd be interesting
print("Sparsicity: ", ((data_dense > 0).sum()/data_dense.size)*100, "%")

# I'll start with a 'stupid' model with no hyperparameter optimization

# Build LDA Model
lda_model = LatentDirichletAllocation(n_topics=200,               # Number of topics
                                      max_iter=10,               # Max learning iterations
                                      learning_method='online',   
                                      random_state=100,          # Random state
                                      batch_size=128,            # n docs in each learning iter
                                      evaluate_every = -1,       # compute perplexity every n iters, default: Don't
                                      n_jobs = -1,               # Use all available CPUs
                                     )
lda_output = lda_model.fit_transform(data_vectorized)

print(lda_model)  # Model attributes

# Log Likelyhood: Higher the better
print("Log Likelihood: ", lda_model.score(data_vectorized))

# Perplexity: Lower the better. Perplexity = exp(-1. * log-likelihood per word)
print("Perplexity: ", lda_model.perplexity(data_vectorized))

# See model parameters
pprint(lda_model.get_params())

#Ok, so simple model is pretty bad. I'll try using GridSearchCV but this takes FOREVER. 
search_params = {'n_components': [50, 100, 250, 500], 'learning_decay': [.3, .5, .7, .9], 'max_iter': [5, 10]}
lda = LatentDirichletAllocation()
model = GridSearchCV(lda, param_grid=search_params)
model.fit(data_vectorized)
