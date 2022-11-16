import string
import numpy as np
import pandas as pd
from os import walk
import re
import gensim
from datetime import datetime
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('punkt')

pd.set_option("display.max_columns", None)

punctuation = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•@'         # define a string of punctuation symbols

# Functions to clean tweets
def remove_links(tweet):
    """Takes a string and removes web links from it"""
    tweet = re.sub(r'http\S+', '', tweet)   # remove http links
    tweet = re.sub(r'bit.ly/\S+', '', tweet)  # remove bitly links
    tweet = tweet.strip('[link]')   # remove [links]
    tweet = re.sub(r'pic.twitter\S+','', tweet)
    return tweet

def remove_users(tweet):
    """Takes a string and removes retweet and @user information"""
    tweet = re.sub('(RT\s@[A-Za-z]+[A-Za-z0-9-_]+)', '', str(tweet))  # remove re-tweet
    tweet = re.sub('(@[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet)  # remove tweeted at
    return tweet

def remove_hashtags(tweet):
    """Takes a string and removes any hash tags"""
    tweet = re.sub('(#[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet)  # remove hash tags
    return tweet

def remove_av(tweet):
    """Takes a string and removes AUDIO/VIDEO tags or labels"""
    tweet = re.sub('VIDEO:', '', tweet)  # remove 'VIDEO:' from start of tweet
    tweet = re.sub('AUDIO:', '', tweet)  # remove 'AUDIO:' from start of tweet
    return tweet

def tokenize(tweet):
    """Returns tokenized representation of words in lemma form excluding stopwords"""
    result = []
    for token in gensim.utils.simple_preprocess(tweet):
        if token not in gensim.parsing.preprocessing.STOPWORDS \
                and len(token) > 2:  # drops words with less than 3 characters
            result.append(lemmatize(token))
    return result

def lemmatize(token):
    """Returns lemmatization of a token"""
    return WordNetLemmatizer().lemmatize(token, pos='v')

def remove_single_word(tweet):
    return '' if len(tweet.split()) < 2 else tweet

def preprocess_tweet(tweet):
    """Main master function to clean tweets, stripping noisy characters, and tokenizing use lemmatization"""
    tweet = remove_users(tweet)
    tweet = remove_links(tweet)
    tweet = remove_hashtags(tweet)
    tweet = remove_av(tweet)
    tweet = tweet.lower()  # lower case
    tweet = re.sub('[' + punctuation + ']+', ' ', tweet)  # strip punctuation
    tweet = re.sub('\s+', ' ', tweet)  # remove double spacing
    tweet = re.sub('([0-9]+)', '', tweet)  # remove numbers
    tweet = re.sub('📝 …', '', tweet)
    tweet_token_list = tokenize(tweet)  # apply lemmatization and tokenization
    tweet = ' '.join(tweet_token_list)
    tweet = remove_single_word(tweet)
    return tweet

def tokenize_tweets(df):

    print('Loading data. Number of Tweets : {}'.format(len(df)))
    df['tweet'] = df.tweet.apply(preprocess_tweet)
    # df['tweet'] = df.tweet.apply(preprocess_tweet)
    df = df.drop_duplicates(subset=['tweet'], keep='last')
    num_tweets = len(df)
    print('Number of Tweets that have been cleaned and tokenized : {}'.format(num_tweets))

    return df


def is_number(w):
    try:
        w = float(w)
        return True
    except:
        pass
        return False


def preprocess(text):
    text = text.lower()
    text = preprocess_tweet(text)
    text_p = ''.join([char for char in text if char not in string.punctuation])

    words = word_tokenize(text_p)

    stop_words = stopwords.words('english')
    filtered_words = [word for word in words if word not in stop_words]

    porter = PorterStemmer()
    stemmed = [porter.stem(word) for word in filtered_words if not is_number(word)]

    return words, filtered_words, stemmed

def applyHelper(x):
    words, filtered_words, stemmed = preprocess(x)
    return stemmed

def yearMonth(y, m):
    if m < 10:
        return str(y) + "0" + str(m)
    else:
        return str(y) + str(m)

### online tweet preprocess ###

dir = 'data/sorted/'
filenames = next(walk(dir), (None, None, []))[2]  # [] if no file

for f in filenames:
    path = dir + f
    df = pd.read_csv(path, encoding="utf-8")
    df = df[['tweet', 'date', 'vax_label']]
    df['date'] = df['date'].apply(lambda x: datetime.fromisoformat(x))
    df['yr_month'] = df['date'].apply(lambda x: yearMonth(x.year, x.month))
    print(df['yr_month'])

    df = df.sort_values(by=['date'], ignore_index=True)
    df.to_csv('data/sorted/'+f, index=False)

    df['tweet'] = df['tweet'].apply(lambda x: applyHelper(str(x)))
    df = df[df["tweet"].str.len() != 0]

    df.to_csv('data/preprocessed/'+f, index=False)

print("online completed")

### offline news preprocess ###
dir = 'data/off_data/crawled/'
filenames = next(walk(dir), (None, None, []))[2]  # [] if no file

filenames = ['news_2020_09.csv', 'news_2021_09.csv']
for f in filenames:
    month = int(f[-6:-4]) # extract month from filename
    df = pd.read_csv(dir+f)
    df.drop(columns=['date'], inplace=True)
    df.rename(columns={'datetime': 'date'}, inplace=True)  # 'title', 'content','summary'
    df['date'] = df['date'].apply(lambda x: datetime.fromisoformat(x))
    df['month'] = df['date'].apply(lambda x: int(x.month))
    df['yr_month'] = df['date'].apply(lambda x: yearMonth(x.year, x.month))
    df = df.loc[df['month'] == month]
    df = df.sort_values(by=['date'], ignore_index=True)
    df['content'] = df['content'].apply(lambda x: applyHelper(x))
    df['title'] = df['title'].apply(lambda x: applyHelper(str(x)))
    df['summary'] = df['summary'].apply(lambda x: applyHelper(x))
    df.to_csv(dir+'preprocessed/'+f, index=False)

