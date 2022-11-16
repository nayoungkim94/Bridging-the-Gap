import numpy as np
import scipy.io
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from gensim.corpora import Dictionary
import time
import sys
from scipy.stats import entropy
import pickle
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy import linalg as LA
from gensim.models.coherencemodel import CoherenceModel
import gensim
from utils import on_off_com, display_words, display_topics, com_diff_score


def PD(n_topic, n_c, steps):

    rank = n_topic   ## number of topics, default=20
    n_d = rank - n_c  # number of different topics

    on_file = '01_pro.csv'
    type = on_file[:-4]
    off_type = 'content' # 'title', 'content','summary'

    # on_file = 'eng_distilbert_clean_10000_1.csv'
    f = 'PD'+off_type+str(n_c)+str(n_d)
    filename = open("results/"+type+"/"+str(steps)+"/"+f+".txt", 'w')
    # words_outfile = open('results/'+type+"/"+str(steps)+"/"+'topics_'+f+'.txt', 'w', encoding='UTF-8')
    tolerance = 1e-4
    myeps = 1e-16
    #alpha_list=[0.1,1,10,100,500,1000]
    #beta_list=[0.1,1,10,100,500,1000]  ##1 is the best
    beta = 0.1
    alpha = 10
    # data_ori = pd.read_csv('cleanHarveyTweets.csv')
    data_ori = on_off_com(on_file, off_type)
    data = data_ori['tweet'].values.tolist()
    print(f"Data size: {len(data_ori.index)}")
    data = data_ori['tweet'].values.tolist()
    dates = data_ori['date'].values.tolist()
    # months = data_ori['date'].apply(lambda x: findMonth(x)).values.tolist()
    typeMonth = data_ori['typeMonth'].values.tolist()

    new_topic_start_date = 'off202001'
    month_idx = data_ori.index[data_ori['typeMonth'] == new_topic_start_date].tolist()
    n_ori = month_idx[0]


    vectorizer = TfidfVectorizer(max_df=0.95, min_df=10,max_features=3000, stop_words='english')
    X = vectorizer.fit_transform(data_ori['tweet'].values.astype('U'))##data size*

    #scipy.io.savemat('Harvey.mat', mdict={'A': X})
    corpus_vect_gensim = gensim.matutils.Sparse2Corpus(X, documents_columns=False)
    dictionary = Dictionary.from_corpus(corpus_vect_gensim,
                                        id2word=dict((id, word) for word, id in vectorizer.vocabulary_.items()))
    tweets=[]
    X = X.toarray()
    for t in data:
        tweets.append(t.split(' '))
    corpus=[dictionary.doc2bow(text) for text in tweets]
    N = X.shape[0]
    feature_name = vectorizer.get_feature_names()
    fea_index = vectorizer.vocabulary_


    #common_list_final = []
    #differ_list_final = []
    fig_num=0
    H_list=[]
    W_list=[]
    HU_list=[]

    n_d = rank-n_c
    alpha_temp = alpha
    # filename.write('n_c: ' + str(n_c)+'\n')
    recon_error = []
    common_list = []
    differ_list = []

    X_df = pd.DataFrame(X)
    X_df['typeMonth'] = typeMonth

    X_stream = X_df[n_ori:]
    X_old = X_df[:n_ori]
    time_list=[]
    #print "current data size %d" % n_ori
    start_time=time.time()
    #W_list.append(W)
    # for g, df in X_stream.groupby(np.arange(len(X_stream)) // new_data_size):

    for _, df in X_stream.groupby('typeMonth', sort=False):
        print(set(df['typeMonth'].values.tolist()))
        df = df.drop(columns=['typeMonth'])
        epoch = 0
        optgap = 1000
        V = X_old.loc[:, X_old.columns != 'typeMonth'].values
        shape = V.shape

        old_n = shape[0]
        new_n = df.shape[0]
        n = old_n + new_n
        print(f"current data size {n} (+{new_n})")

        U = df.values

        H = np.random.rand(rank, shape[1]).astype(np.float64)
        H = np.divide(H, H.max())

        W = np.random.rand(shape[0], rank).astype(np.float64)
        W = np.divide(W, W.max())

        Wc = W[:, 0:n_c]
        Wd = W[:, n_c:]
        Hc = H[0:n_c, :]
        Hd = H[n_c:, :]

        HU = np.random.rand(rank, shape[1]).astype(np.float64)
        HU = np.divide(HU, HU.max())

        WU = np.random.rand(U.shape[0], rank).astype(np.float64)
        WU = np.divide(WU, WU.max())
        WUc = WU[:, 0:n_c]
        WUd = WU[:, n_c:]

        HUc = HU[0:n_c, :]
        HUd = HU[n_c:, :]
        cost_last=(LA.norm(V - np.matmul(W, H), ord='fro'))**2/old_n+(LA.norm(U- np.matmul(WU, HU), ord='fro'))**2/new_n+ alpha * (LA.norm( Hc- HUc,2)) ** 2 \
                  + beta * np.sum(np.abs(np.matmul(Hd.T, HUd)))

        while epoch < steps and optgap > tolerance:
            epoch+=1
            temp = np.matmul(WUd.T, U)*2/new_n
            denom = np.matmul(np.matmul(WUd.T, WU), HU)*2/new_n
            denom = denom + beta * np.matmul(Hd, np.sign(np.matmul(Hd.T,HUd)))+myeps
            HUd = HUd * temp / (denom + myeps)
            HUd[HUd < myeps] = myeps

            temp = np.matmul(WUc.T, U)*2/new_n + 2*alpha * Hc
            denom = np.matmul(np.matmul(WUc.T, WU),
                              np.concatenate([HUc, HUd], 0))*2/new_n + 2*alpha* HUc
            HUc = HUc * temp / (denom + myeps)
            HUc[HUc < myeps] = myeps

            temp = np.matmul(U, HUd.T)*2/new_n
            denom = np.matmul(np.matmul(np.concatenate([WUc, WUd], 1), np.concatenate([HUc, HUd], 0)), HUd.T)*2/new_n
            WUd = WUd * temp / (denom + myeps)
            WUd[WUd < myeps] = myeps

            temp = np.matmul(U, HUc.T)*2/new_n
            denom = np.matmul(np.matmul(np.concatenate([WUc, WUd], 1), np.concatenate([HUc, HUd], 0)), HUc.T)*2/new_n
            WUc = WUc * temp / (denom + myeps)
            WUc[WUc < myeps] = myeps

            ###############################################################################
            temp = np.matmul(V, Hd.T)*2/old_n
            denom = np.matmul(np.matmul(np.concatenate([Wc, Wd], 1), np.concatenate([Hc, Hd], 0)), Hd.T)*2/old_n
            Wd = Wd * temp / (denom + myeps)
            Wd[Wd < myeps] = myeps

            temp = np.matmul(V, Hc.T)*2/old_n
            denom = np.matmul(np.matmul(np.concatenate([Wc, Wd], 1), np.concatenate([Hc, Hd], 0)), Hc.T)*2/old_n
            Wc = Wc * temp / (denom + myeps)
            Wc[Wc < myeps] = myeps

            temp = np.matmul(Wd.T, V)*2/old_n
            denom = np.matmul(np.matmul(Wd.T, np.concatenate([Wc, Wd], 1)), np.concatenate([Hc, Hd], 0))*2/old_n
            denom = denom + beta * np.matmul(HUd,np.sign(np.matmul(HUd.T, Hd))) + myeps
            Hd = Hd * temp / (denom + myeps)
            Hd[Hd < myeps] = myeps

            temp = np.matmul(Wc.T, V)*2/old_n + 2*alpha * HUc
            denom = np.matmul(np.matmul(Wc.T, np.concatenate([Wc, Wd], 1)),
                              np.concatenate([Hc, Hd], 0)) + 2*alpha * Hc
            Hc = Hc * temp / (denom + myeps)
            Hc[Hc < myeps] = myeps

            H = np.concatenate([Hc, Hd], 0)
            W = np.concatenate([Wc, Wd], 1)
            HU = np.concatenate([HUc, HUd], 0)
            WU = np.concatenate([WUc, WUd], 1)
            cost = (LA.norm(V - np.matmul(W, H), ord='fro')) ** 2 / old_n + (
                LA.norm(U - np.matmul(WU, HU), ord='fro')) ** 2 / new_n + alpha * (
                            LA.norm(Hc - HUc, 2)) ** 2 \
                        + beta * np.sum(np.abs(np.matmul(Hd.T, HUd)))
            optgap = abs(cost_last - cost) / cost_last
            cost_last = cost
            # print("This is the epoch %d, the cost is %f" % (epoch, cost))

        time_list.append(time.time() - start_time)
        #alpha_temp=alpha_temp/np.sqrt(g+1) ###parameter decay, square root decay
        H_list.append(H)
        W_list.append(W)
        HU_list.append(HU)
        recon_error.append((LA.norm(U - np.matmul(WU, HU), ord='fro')) ** 2 / n)
        common, differece = com_diff_score(H, HU, n_c, n_d)
        n = old_n + new_n
        # print(common)
        # print(differece)
        # print(recon_error)
        new_data = X_df[:n]
        X_old = new_data

    ###########calculate common/different scores################
    for i in range(len(H_list)):
        H1 = H_list[i]
        H2 = HU_list[i]
        common, differece = com_diff_score(H1, H2, n_c, n_d)
        common_list.append(common)
        differ_list.append(differece)

    filename.write("time: "+' '.join(map(str, time_list)) + '\n')#smaller
    filename.write("common: "+' '.join(map(str, common_list)) + '\n')  #smaller
    filename.write("differ: "+' '.join(map(str, differ_list)) + '\n')  # ###bigger
    filename.write("error: "+' '.join(map(str, recon_error)) + '\n')  #  ###smaller


    filename.close()


n_topics = range(5, 10, 5)
steps = range(100, 500, 100)

# for s in steps:
#     for t in n_topics:
#         c = int(t/2)
#         PD(t, c, s)

PD(5, 2, 100)

