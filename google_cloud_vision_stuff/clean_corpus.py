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