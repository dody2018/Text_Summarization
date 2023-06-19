#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from gensim.summarization import keywords
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

# Download packages (run only once)
# nltk.download('punkt')
# nltk.download('stopwords')

# Preprocessing
text = 'my name is ghada moataz'
sentences = nltk.sent_tokenize(text)
words = [nltk.word_tokenize(sentence) for sentence in sentences]
stop_words = set(stopwords.words('english'))
words_filtered = [[word for word in sentence if not word.lower() in stop_words] for sentence in words]
porter = PorterStemmer()
words_stemmed = [[porter.stem(word) for word in sentence] for sentence in words_filtered]

# Sentence scoring with TextRank
graph = nx.Graph()
for i, sentence_i in enumerate(words_stemmed):
    for j, sentence_j in enumerate(words_stemmed):
        if i != j:
            similarity = cosine_similarity([sentence_i], [sentence_j])[0][0]
            if similarity > 0:
                graph.add_edge(i, j, weight=similarity)
scores = nx.pagerank(graph)

# Extract the top N most important sentences
num_sentences = 3
top_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)[:num_sentences]

# Generate the summary
summary = ' '.join([sentence for score, sentence in top_sentences])
print(summary)

