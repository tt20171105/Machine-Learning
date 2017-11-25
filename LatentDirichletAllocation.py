# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 01:52:53 2017

@author: tt20171105
"""
import re
import MeCab
import numpy as np
import pandas as pd
from pprint import pprint
from gensim import corpora, models, similarities

INPUT_PATH = ""
OUT_PATH   = ""

def getnoun(_word):
    noun   = ""
    tagger = MeCab.Tagger( "-Ochasen" )
    node   = tagger.parseToNode("")    #おまじない
    node   = tagger.parseToNode(_word)
    while node:
        replace_node = re.sub(re.compile("[!-/:-@[-`{-~]"), "", node.surface)
        _node = node
        node  = node.next
        if _node.feature.split(",")[0] != "名詞": continue  #名詞でなければ飛ばす
        if (replace_node == "") or \
           (replace_node == " ") or \
           (len(replace_node) == 1): continue  #空文字、空白、一文字は飛ばす
        if replace_node.isdigit()  : continue  #数字は飛ばす
        noun += replace_node + " "  #分かち書きにする
    return noun

def lda(_texts, num_topics, tfidf=True, save=True, load=False):
    if load:
        #dictionaryのロード
        dictionary = corpora.Dictionary.load_from_text(OUT_PATH + 'dictionary.dict')
        #コーパスのロード
        corpus = corpora.MmCorpus(OUT_PATH + 'corpus.mm')
        #作成済みldaモデルを使用する
        model  = models.LdaModel.load(OUT_PATH + 'lda.model')
    else:
        #単語ベクトルを作成する
        dictionary = corpora.Dictionary(_texts)
        corpus     = [dictionary.doc2bow(text) for text in _texts]
        #tfidfコーパスを使用する
        if tfidf:
            tfidf   = models.TfidfModel(corpus)
            _corpus = tfidf[corpus]
        else:
            _corpus = corpus
        #lda
        model = models.LdaModel(corpus=_corpus, id2word=dictionary,
                                num_topics=num_topics, minimum_probability=0.001,
                                passes=20, update_every=0, chunksize=10000)
        #保存する
        if save:
            #dictionaryの保存
            dictionary.save_as_text(OUT_PATH + 'dictionary.dict')
            #コーパスの保存
            corpora.MmCorpus.serialize(OUT_PATH + 'corpus.mm', corpus)
            #モデルの保存
            model.save(OUT_PATH + 'lda.model')
    
    return dictionary, corpus, model

def create_data(filename):
    df = pd.read_csv(INPUT_PATH + filename, encoding="cp932", names=["word"])
    #形態素解析
    df["noun"] = df["word"].apply(lambda x: getnoun(x))
    #1行を1つのドキュメントとして扱う
    texts      = list(df["noun"].apply(lambda x: x.split(" ")))
    #全ての単語をリストに入れる
    all_words  = list(df["noun"].sum().split(" "))
    #一度しか出現していない単語は除く
    once_word  = set(w for w in set(all_words) if all_words.count(w)==1)
    texts_not_once_word = [[w for w in t if w not in once_word] for t in texts]

    texts = []
    for i in range(len(texts_not_once_word)):
        if texts_not_once_word[i] == []   : continue  #空のリストは除く
        if texts_not_once_word[i] in texts: continue  #同じドキュメントは除く
        texts.append(texts_not_once_word[i])
    print("目安", int(np.sqrt(len(set(all_words)))/2))
    return texts

#LDAモデルを作成する
texts = create_data("")
dictionary, corpus, model = lda(texts, 7, tfidf=True, save=True, load=False)
pprint(model.show_topics())

#比較対象の文章
texts_compared = create_data("")
texts_compared = [x for y in texts_compared for x in y]
texts_compared = set(texts_compared)
vec_compared   = dictionary.doc2bow(texts_compared)
#類似topicを算出する
model_compared = model[vec_compared]

#コサイン類似度を算出する
lda_index = similarities.MatrixSimilarity(model[corpus])
lda_sims  = lda_index[model_compared]
max_num   = 10
print(sorted(enumerate(lda_sims), key=lambda item: -item[1])[:max_num])
for elem in sorted(enumerate(lda_sims), key=lambda item: -item[1])[:max_num]:
    print(u'類似度=' + str(elem[1]) + ': ' + "、".join(texts[elem[0]]))
