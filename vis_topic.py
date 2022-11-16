import matplotlib.pyplot as plt
from matplotlib import cm
import gensim
from gensim.models import Word2Vec
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import pandas as pd
import numpy as np
import nltk
# nltk.download('punkt')

from itertools import chain
import torch
from transformers import BertTokenizer, BertModel
from sklearn.manifold import TSNE

path = 'results/topics_on+off1010.txt'



def rowIndex(row):
    return row.name

def getKey(dct, value):
    return [key for key in dct if (value in dct[key])]


def topic_dict(path):
    num = ""
    for c in path:
        if c.isdigit():
            num = num + c
    l = len(num)
    num_common=int(num[:int(l/2)])
    num_distinct=int(num[int(l/2):])
    file = open(path, 'r')
    lines = file.readlines()

    topics = {}
    cur = ''
    key = ''
    for line in lines:
        line = line.strip('\n')
        alist = line.split(' ')

        if ('tweet' in alist or 'news' in alist) and len(alist) == 4 and "~" in alist:
            tmp = alist[3] + alist[0][:7] # type-yr-mo
            cur = tmp
        elif 'Topic' in alist :
            if int(alist[1][:-1]) <= num_common: # common topics
                tmp = "C" + alist[1][:-1]
                key = cur+tmp
            elif int(alist[1][:-1]) > num_common: # distinct topics
                tmp = "D" + alist[1][:-1]
                key = cur+tmp
        elif "##################" not in alist and alist!=['']:
            if key != '':
                topics[key] = alist

    return topics


def word2vec(path):
    topics = topic_dict(path)
    # topic_num = len(topics['tweet2020-01C1'])

    tokens = topics.values()
    flatten_tokens = list(chain.from_iterable(tokens))
    model = Word2Vec(tokens, vector_size=50, sg=1, min_count=1)
    # model.save("word2vec.model")
    # model = Word2Vec.load("word2vec.model")

    wv = model.wv
    # wv.save("word2vec.wordvectors")
    # wv = KeyedVectors.load("word2vec.wordvectors", mmap='r')

    vocab = [wv.index_to_key[i] for i in range(len(wv))]
    X = wv.vectors

    return X, vocab, topics

def PCA(path):
    # topic_df = pd.DataFrame(topics)
    # topic_df = topic_df.applymap(lambda x: wv[x])
    word_vectors, vocab, topics = word2vec(path)
    df = pd.DataFrame(word_vectors)
    df.rename(index=lambda x: vocab[x], inplace=True)

    #Computing the correlation matrix
    X_corr = df.corr()

    #Computing eigen values and eigen vectors
    values, vectors = np.linalg.eig(X_corr)

    #Sorting the eigen vectors coresponding to eigen values in descending order
    args = (-values).argsort()
    values = vectors[args]
    vectors = vectors[:, args]

    #Taking first 2 components which explain maximum variance for projecting
    new_vectors = vectors[:, :2]

    #Projecting it onto new dimesion with 2 axis
    neww_X = np.dot(word_vectors, new_vectors)
    neww_df = pd.DataFrame(neww_X, index=vocab)
    neww_df['topic'] = neww_df.apply(rowIndex, axis=1)
    neww_df['topic'] = neww_df['topic'].apply(lambda x: getKey(topics, x))
    neww_df = neww_df.explode('topic')

    g1 = neww_df.loc[neww_df['topic'].str.contains('tweet2020-01C')]
    g2 = neww_df.loc[neww_df['topic'].str.contains('news2020-01D')]
    g3 = neww_df.loc[neww_df['topic'].str.contains('tweet2020-02C')]
    g4 = neww_df.loc[neww_df['topic'].str.contains('tweet2020-02D')]


    # g1.drop(columns='topic', inplace=True)
    plt.figure(figsize=(13, 7))
    plt.scatter(g1[0].values.tolist(), g1[1].values.tolist(), linewidths=1, color='blue')
    plt.scatter(g2[0].values.tolist(), g2[1].values.tolist(), linewidths=1, color='red')
    plt.scatter(g3[0].values.tolist(), g3[1].values.tolist(), linewidths=1, color='green')
    plt.scatter(g4[0].values.tolist(), g4[1].values.tolist(), linewidths=1, color='purple')


    # plt.scatter(neww_X[:, 0], neww_X[:, 1], linewidths=5, c=neww_df['topic'].map(colors))
    plt.xlabel("PC1", size=15)
    plt.ylabel("PC2", size=15)
    plt.title("Word Embedding Space", size=20)
    # for i, word in enumerate(vocab):
    #   plt.annotate(word, xy=(neww_X[i,0],neww_X[i,1]))

    plt.show()

# PCA(path)


def tSNE2d(path):
    word_vectors, vocab, topics = word2vec(path)
    print(word_vectors.shape)
    df = pd.DataFrame(word_vectors)
    df.rename(index=lambda x: vocab[x], inplace=True)

    # Create a two dimensional t-SNE projection of the embeddings
    tsne = TSNE(2, verbose=1)
    tsne_proj = tsne.fit_transform(word_vectors)
    neww_df = pd.DataFrame(tsne_proj, index=vocab)
    neww_df['topic'] = neww_df.apply(rowIndex, axis=1)
    neww_df['topic'] = neww_df['topic'].apply(lambda x: getKey(topics, x))
    neww_df = neww_df.explode('topic')

    cmap = cm.get_cmap('tab20')
    g1 = neww_df.loc[neww_df['topic'].str.contains('tweet2020-01C')]
    g2 = neww_df.loc[neww_df['topic'].str.contains('news2020-01D')]
    g3 = neww_df.loc[neww_df['topic'].str.contains('tweet2020-02C')]
    g4 = neww_df.loc[neww_df['topic'].str.contains('tweet2020-02D')]

    # g1.drop(columns='topic', inplace=True)
    plt.figure(figsize=(13, 7))
    plt.scatter(g1[0].values.tolist(), g1[1].values.tolist(), linewidths=1, color='blue')
    plt.scatter(g2[0].values.tolist(), g2[1].values.tolist(), linewidths=1, color='red')
    plt.scatter(g3[0].values.tolist(), g3[1].values.tolist(), linewidths=1, color='green')
    plt.scatter(g4[0].values.tolist(), g4[1].values.tolist(), linewidths=1, color='purple')

    # plt.xlabel("PC1", size=15)
    # plt.ylabel("PC2", size=15)
    plt.title("Word Embedding Space", size=20)
    # Plot those points as a scatter plot and label them based on the pred labels
    # cmap = cm.get_cmap('tab20')
    # fig, ax = plt.subplots(figsize=(8,8))
    num_categories = 10
    # ax.scatter(tsne_proj[indices,0],tsne_proj[indices,1], c=np.array(cmap(lab)).reshape(1,4), label = lab ,alpha=0.5)
    # ax.legend(fontsize='large', markerscale=2)
    plt.show()


tSNE2d(path)