
import glob
import os
import re
import pickle
import nltk
import numpy as np
import datetime
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from sys import argv
from nltk.corpus import stopwords
stop = stopwords.words('english')
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def cleanse(article):

    article = re.sub('[^A-Za-z .-]+', ' ', article)
    article = article.replace('-', '')
    article = article.replace('...', '')
    article = article.replace('Mr.', 'Mr').replace('Mrs.', 'Mrs')

    article = remove_acronyms(article)

    article = ' '.join(article.split())
    return article

def remove_stop_words(article):
    article = ' '.join([i for i in article.split() if i not in stop])
    return article


def remove_acronyms(s):
    r = re.compile(r'(?:(?<=\.|\s)[A-Z]\.)+')
    acronyms = r.findall(s)
    for a in acronyms:
        s = s.replace(a, a.replace('.',''))
    return s

def sen_rank(doc, doc_matrix, feature_names, top_n=3):
    sents = nltk.sent_tokenize(doc)
    sentences = [nltk.word_tokenize(sent) for sent in sents]
    
    tfidf_sent = [[doc_matrix[feature_names.index(w.lower())]
                   for w in sent if w.lower() in feature_names]
                 for sent in sentences]

    doc_val = sum(doc_matrix)
    sent_values = [sum(sent) / doc_val for sent in tfidf_sent]

    
    ranked_sents = [sent*(i/len(sent_values))
                    for i, sent in enumerate(sent_values)]

    ranked_sents = [pair for pair in zip(range(len(sent_values)), sent_values)]
    ranked_sents = sorted(ranked_sents, key=lambda x: x[1] *-1)

    return ranked_sents[:top_n]

if __name__ == '__main__':
    
	file_list = glob.glob(os.path.join(os.getcwd(),argv[1], "*.txt"))
	data = []
	for file_path in file_list:
	     with open(file_path,encoding='utf8') as f_input:
	          data.append(f_input.read())
	f = open(argv[2], "r")
	article = f.read()
	cleaned_article = cleanse(article)
	doc = remove_stop_words(cleaned_article)
	data = [' '.join(article) for article in data]
	train_data = set(data + [doc])
	
	vectorizer = CountVectorizer()
	vectorizer = vectorizer.fit(train_data)
	freq_term_matrix = vectorizer.transform(train_data)
	feature_names = vectorizer.get_feature_names()
	
	tfidf = TfidfTransformer(norm="l2")
	tfidf.fit(freq_term_matrix)
	
	story_freq_term_matrix = vectorizer.transform([doc])
	story_tfidf_matrix = tfidf.transform(story_freq_term_matrix)
	story_dense = story_tfidf_matrix.todense()
	doc_matrix = story_dense.tolist()[0]
	
	top_sen = sen_rank(doc, doc_matrix, feature_names)
	summary = '.'.join([cleaned_article.split('.')[i]
                        for i in [pair[0] for pair in top_sen]])
	summary = ' '.join(summary.split())
	print(summary)