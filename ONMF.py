import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
import sys
import scipy.io
import time
import pickle
from numpy import linalg as LA
from utils import on_off_com, display_words, display_topics, com_diff_score




def ONMF(n_topic, n_c, steps):

    rank = n_topic   ## number of topics, default=20
    n_d = rank - n_c  # number of different topics
    tolerance = 1e-4
    myeps = 1e-16

    on_file = '01_anti.csv'
    type = on_file[:-4]
    off_type = 'content' # 'title', 'content','summary'
    # on_file = 'eng_distilbert_clean_10000_1.csv'
    f = 'ONMF'+off_type+str(n_c)+str(n_d)
    filename = open("results/"+type+"/"+str(steps)+"/"+f+".txt", 'w')
    words_outfile = open('results/'+type+"/"+str(steps)+"/"+'topics_'+f+'.txt', 'w', encoding='UTF-8')




    data = on_off_com(on_file, off_type)
    # convert the text to a tf-idf weighted term-document matrix
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=10,max_features=3000, stop_words='english')
    X = vectorizer.fit_transform(data['tweet'].values.astype('U'))##data size*


    corpus_vect_gensim = gensim.matutils.Sparse2Corpus(X, documents_columns=False)
    dictionary = Dictionary.from_corpus(corpus_vect_gensim,
                                        id2word=dict((id, word) for word, id in vectorizer.vocabulary_.items()))

    X = X.toarray()
    N = X.shape[0]
    tweets = []
    for t in data:
        tweets.append(t.split(' '))
    corpus = [dictionary.doc2bow(text) for text in tweets]

    feature_name = vectorizer.get_feature_names()

    #scipy.io.savemat('data_5000.mat', mdict={'A': X})


    dates = data['date'].values.tolist()
    # months = data_ori['date'].apply(lambda x: findMonth(x)).values.tolist()
    typeMonth = data['typeMonth'].values.tolist()

    new_topic_start_date = 'off202001'
    month_idx = data.index[data['typeMonth'] == new_topic_start_date].tolist()
    n_ori = month_idx[0]

    recon_error = []
    common_list = []
    differ_list = []
    timestamp = []

    X_df = pd.DataFrame(X)
    X_df['typeMonth'] = typeMonth
    X_stream = X_df[n_ori:]
    X_old = X_df[:n_ori]
    V = X_old.loc[:, X_old.columns != 'typeMonth'].values
    shape = V.shape

    start_date = data['date'].iloc[0]
    end_date = data['date'].iloc[n_ori - 1]
    print(f"start time period {start_date} ~ {end_date}")
    timestamp.append((start_date, end_date))


    ####initialize
    H = np.random.rand(rank, shape[1]).astype(np.float64)
    H = np.divide(H, H.max())

    W = np.random.rand(shape[0], rank).astype(np.float64)
    W = np.divide(W, W.max())

    cost_last = (LA.norm(V - np.matmul(W, H), ord='fro')) ** 2 / n_ori

    #print cost_last

    ######main loop
    epoch = 0
    optgap = 1000
    time_list = []
    H_list = []
    RE = []
    HU_list = []
    W_list = []

    print("current data size %d" % n_ori)
    start_time = time.time()

    while epoch < steps and optgap > tolerance:
        ##########update H
        WTV = np.matmul(W.T, V)
        WTWH = np.matmul(np.matmul(W.T, W), H)
        H = H * WTV / WTWH
        H = H + myeps*(H < myeps)

        ##########update W
        VHT = np.matmul(V, H.T)
        WHHT = np.matmul(W, np.matmul(H, H.T))
        W = W * VHT / WHHT
        W = W + myeps * (W < myeps)

        epoch += 1
        cost = (LA.norm(V - np.matmul(W, H), ord='fro')) ** 2 / n_ori
        optgap = abs(cost_last - cost)/cost_last
        cost_last = cost
        # print("This is the epoch %d, the cost is %f" % (epoch, cost))

    time_list.append(time.time()-start_time)
    H_list.append(H)
    W_list.append(W)

    subset_size = [n_ori]

    for _, df in X_stream.groupby('typeMonth', sort=False):
        print(set(df['typeMonth'].values.tolist()))
        df = df.drop(columns=['typeMonth'])

        epoch = 0
        optgap = 1000
        old_n = X_old.shape[0]
        new_n = df.shape[0]
        n = old_n + new_n
        print(f"current data size {n} (+{new_n})")
        subset_size.append(new_n)
        new_data = X_df[:n]
        U = df.values

        start_date = data['date'].iloc[old_n]
        end_date = data['date'].iloc[n - 1]
        print(f"current time period {start_date} ~ {end_date}")
        print("-------------------------------------------")
        timestamp.append((start_date, end_date))

        start_time = time.time()
        HU = np.random.rand(rank, shape[1]).astype(np.float64)
        HU = np.divide(HU, HU.max())

        WU = np.random.rand(new_n, rank).astype(np.float64)
        WU = np.divide(WU, WU.max())
        cost_last = (LA.norm(U - np.matmul(WU, HU), ord='fro')) ** 2 / new_n

        while epoch < steps and optgap > tolerance:
            ##########update H
            WTV = np.matmul(WU.T, U)
            WTWH = np.matmul(np.matmul(WU.T, WU), HU)
            HU = HU * WTV / WTWH
            HU = HU + myeps * (HU < myeps)

            ##########update W
            VHT = np.matmul(U, HU.T)
            WHHT = np.matmul(WU, np.matmul(HU, HU.T))
            WU = WU * VHT / WHHT
            WU = WU + myeps * (WU < myeps)

            epoch += 1
            cost = (LA.norm(U - np.matmul(WU, HU), ord='fro')) ** 2 / n
            optgap = abs(cost_last - cost) / cost_last
            cost_last = cost
            # print("This is the epoch %d, the cost is %f" % (epoch, cost))

        HU_list.append(HU)
        epoch = 0
        optgap = 1000
        diag_mat = np.zeros([rank, rank])

        for i in range(rank):
            diag_mat[i, i] = LA.norm(W[:, i])
        deltaV=np.concatenate([np.matmul(diag_mat,H),U],0)
        shape=deltaV.shape
        deltaH = np.abs(np.random.rand(rank, shape[1]).astype(np.float64))
        deltaH = np.divide(deltaH, deltaH.max())

        deltaW = np.abs(np.random.rand(rank + U.shape[0], rank).astype(np.float64))
        deltaW = np.divide(deltaW, deltaW.max())
        ################get the temporal H and W
        cost_last=(LA.norm(deltaV - np.matmul(deltaW, deltaH), ord='fro'))**2/2
        #print cost_last
        while epoch < steps and optgap > tolerance:
            WTV = np.matmul(deltaW.T, deltaV)
            WTWH = np.matmul(np.matmul(deltaW.T, deltaW), deltaH)
            deltaH = deltaH*WTV/WTWH
            deltaH = deltaH + myeps * (deltaH < myeps)

            ##########update W
            VHT = np.matmul(deltaV, deltaH.T)
            WHHT = np.matmul(np.matmul(deltaW, deltaH), deltaH.T)
            deltaW = deltaW*VHT/WHHT
            deltaW = deltaW + myeps * (deltaW < myeps)

            epoch += 1
            cost = (LA.norm(deltaV - np.matmul(deltaW, deltaH), ord='fro'))**2/2
            optgap = abs(cost_last - cost) / cost_last
            cost_last = cost
            #print "This is the epoch %d" % epoch

        time_list.append(time.time() - start_time)
        W1=deltaW[0:rank,:]
        W2=deltaW[rank:,:]
        RE.append((LA.norm(U - np.matmul(W2, deltaH), ord='fro')) ** 2 / new_n)
        H = deltaH
        W = np.concatenate([np.matmul(np.matmul(W, np.linalg.inv(diag_mat)),W1), W2], 0)
        H_list.append(H)
        W_list.append(W)
        X_old = new_data


    # print(RE)
    # print(time_list)
    with open('data_pkl/ONMF'+str(n_c)+str(n_d)+'.pkl', 'wb') as f1:
        pickle.dump([W_list, H_list, time_list, HU_list], f1)





    #######starts evaluate#######################
    for i in range(len(H_list)):
        H = H_list[i]
        W = W_list[i]
        # pmi_val = []
        if i == 0:
            n = n_ori
        else:
            if temp + subset_size[i] > N:
                n = N
            else:
                n = temp + subset_size[i]
        documents = data[0:n]
        # X_df = X_df.loc[:, X_df.columns!='typeMonth']
        X_new = X_df[:n]
        datatype = 'tweet' if 'on' in str(X_new.loc[n-1, 'typeMonth']) else 'news'
        X_new = X_new.loc[:, X_new.columns!='typeMonth']
        A = X_new.values

        # print(documents)
        words_outfile.write(f"{timestamp[i][0]} ~ {timestamp[i][1]} {datatype}\n\n")
        display_words(H, feature_name, words_outfile)
        words_outfile.write('\n##################\n')
        # print(H)
        # print(feature_name)
        topics = display_topics(H, feature_name)
        CM = CoherenceModel(topics=topics, corpus=corpus, dictionary=dictionary, coherence='u_mass')
        # pmi_list.append(CM.get_coherence())

        recon_error.append((LA.norm(A - np.matmul(W, H), ord='fro')) ** 2 / n)
        temp = n

    words_outfile.close()

    ###########calculate common/different scores################
    for i in range(len(HU_list)):
        H1 = H_list[i]
        H2 = HU_list[i]
        common, differece = com_diff_score(H1, H2, n_c, n_d)
        common_list.append(common)
        differ_list.append(differece)

    filename.write("time: " + ' '.join(map(str, time_list)) + '\n')  # smaller
    filename.write("common: " + ' '.join(map(str, common_list)) + '\n')  # smaller
    filename.write("differ: " + ' '.join(map(str, differ_list)) + '\n')  # ###bigger
    filename.write("error: " + ' '.join(map(str, recon_error)) + '\n')  # ###smaller

    filename.close()


n_topics = range(5, 10, 5)
steps = range(100, 500, 100)

# for s in steps:
#     for t in n_topics:
#         c = int(t/2)
#         ONMF(t, c, s)

ONMF(5, 2, 100)
