# -*- coding: utf-8 -*-
from gensim.models import Word2Vec

en_wiki_word2vec_model = Word2Vec.load('2800a.model')

testwords = ['天地']
for i in range(1):
    res = en_wiki_word2vec_model.most_similar(testwords[i])
    print (testwords[i])
    print (res)