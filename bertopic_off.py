import numpy as np
import pandas as pd
import sys
from os import walk
import torch
from tqdm import tqdm
import pickle
import transformers
from torch.utils.data import DataLoader, Dataset
from bertopic import BERTopic
import nltk
import string
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from collections import defaultdict

nltk.download('punkt')
nltk.download('stopwords')


def is_number(w):
    try:
        w = float(w)
        return True
    except:
        pass
        return False


def preprocess(text):
    text = text.lower()
    text_p = ''.join([char for char in text if char not in string.punctuation])

    words = word_tokenize(text_p)

    stop_words = stopwords.words('english')
    filtered_words = [word for word in words if word not in stop_words]

    porter = PorterStemmer()
    stemmed = [porter.stem(word) for word in filtered_words if not is_number(word)]

    return words, filtered_words, stemmed


off_path = './data/off_data/crawled/preprocessed/'
filenames = next(walk(off_path), (None, None, []))[2]  # [] if no file

off_df = pd.DataFrame()
for f in filenames:
    off_df = pd.concat([off_df, pd.read_csv(off_path + f)], ignore_index=True)

# print(off_df)

titles = off_df['title'].values.tolist()
l = [s for t in titles for s in t]

model = BERTopic(language="english", calculate_probabilities=True, verbose=True, nr_topics='auto')
topics, probs = model.fit_transform(l)
model.save('./saved_bertopic')



model = BERTopic.load('./saved_bertopic')

sol_docs = titles

topics, probs = model.transform(sol_docs)

topics_dict = defaultdict(list)



for i in range(len(topics)):
    topics_dict[ topics[i] ].append(i)

# print the topics top words
for topic_ind, docs in topics_dict.items():
    print(topic_ind, [i[0] for i in model.get_topic(topic_ind)])
    print('......')
    for i in docs:
        print(sol_docs[i])
        print('----')
    print('======\n======')
