from tqdm import tqdm
import sys
import numpy as np
import pandas as pd
import time
from numpy import linalg as LA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gensim
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from utils import on_off_com, display_words, display_topics, com_diff_score, findMonth

pd.set_option('display.max_column', None)




def TDF(n_topic, n_c, steps):

    rank = n_topic   ## number of topics, default=20
    n_d = rank - n_c  # number of different topics


    on_file = '01_anti.csv'
    type = on_file[:-4]
    off_type = 'content'  # 'title', 'content','summary'
    f = 'TDF'+off_type+str(n_c)+str(n_d)
    filename = open("results/"+type+"/"+str(steps)+"/"+f+".txt", 'w')
    words_outfile = open('results/'+type+"/"+str(steps)+"/"+'topics_'+f+'.txt', 'w', encoding='UTF-8')


    # new_data_size = 1000
    # n_c_list =[3,5,7,9] ###number of common topics
    no_top_documents = 10  ### number of tweets to show

    tolerance = 1e-4
    myeps = 1e-16


    # alpha_list=[0.1,1,10,100,500,1000]
    beta_list = [0.1]  ##1 is the best, default=0.1
    # beta=0.1
    alpha = 1000




    ######################


    data_ori = on_off_com(on_file, off_type)
    print(f"Data size: {len(data_ori.index)}")
    data = data_ori['tweet'].values.tolist()
    dates = data_ori['date'].values.tolist()
    months = data_ori['date'].apply(lambda x: findMonth(x)).values.tolist()
    typeMonth = data_ori['typeMonth'].values.tolist()

    new_topic_start_date = 'off202001'
    month_idx = data_ori.index[data_ori['typeMonth'] == new_topic_start_date].tolist()
    n_ori = month_idx[0]
    # print(data_ori[data_ori['typeMonth'] == new_topic_start_date])
    # print(month_idx)

    # new_topic_start_date = '2020-02-01'
    # month_idx = data_ori.index[data_ori['date'] == new_topic_start_date].tolist()




    # common_list_final = []
    # differ_list_final = []
    fig_num = 0
    # n_d=rank-n_c



    vectorizer = TfidfVectorizer(max_df=0.95, min_df=10, max_features=3000, stop_words='english')
    X = vectorizer.fit_transform(data_ori['tweet'].values.astype('U'))  ##data size*


    # scipy.io.savemat('Florence.mat', mdict={'A': X})
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
    fea_index = vectorizer.vocabulary_

    timestamp = []

    for beta in beta_list:
        # print("beta:", beta)
        
        # n_d=rank-n_c
        alpha_temp = alpha
        # filename.write('beta ' + str(beta) + '\n')
        recon_error = []
        common_list = []
        differ_list = []

        X_df = pd.DataFrame(X)
        # X_df['date'] = dates
        # X_df['month'] = months
        X_df['typeMonth'] = typeMonth

        X_stream = X_df[n_ori:]
        X_old = X_df[:n_ori]
        V = X_old.loc[:, X_old.columns!='typeMonth'].values
        shape = V.shape

        start_date = data_ori['date'].iloc[0]
        end_date = data_ori['date'].iloc[n_ori-1]
        print(f"start time period {start_date} ~ {end_date}")
        timestamp.append((start_date, end_date))
        ####initialize
        H = np.random.rand(rank, shape[1]).astype(np.float64)
        H = np.divide(H, H.max())

        W = np.random.rand(shape[0], rank).astype(np.float64)
        W = np.divide(W, W.max())

        cost_last = (LA.norm(V - np.matmul(W, H), ord='fro')) ** 2 / n_ori

        ######main loop
        epoch = 0
        optgap = 1000
        time_list = []
        Htemp_list = []
        H_U = []
        H_list = []
        W_list = []


        # print("current data size %d" % n_ori)
        start_time = time.time()
        while epoch < steps and optgap > tolerance:
            ##########update H
            WTV = np.matmul(W.T, V)
            WTWH = np.matmul(np.matmul(W.T, W), H)
            H = H * WTV / (WTWH + myeps)
            H[H < myeps] = myeps

            ##########update W
            VHT = np.matmul(V, H.T)
            WHHT = np.matmul(W, np.matmul(H, H.T))
            W = W * VHT / (WHHT + myeps)
            W[W < myeps] = myeps
            epoch += 1
            cost = (LA.norm(V - np.matmul(W, H), ord='fro')) ** 2 / n_ori
            # print cost
            optgap = abs(cost_last - cost)
            cost_last = cost
            # print "This is the epoch %d, the cost is %f" % (epoch,cost)

        time_list.append(time.time() - start_time)
        H_list.append(H)
        W_list.append(W)

        subset_size = [n_ori]



        # batch = []
        for _, df in X_stream.groupby('typeMonth', sort=False):
            print(set(df['typeMonth'].values.tolist()))
            df = df.drop(columns=['typeMonth'])
            epoch = 0
            optgap = 1000
            old_n = X_old.shape[0]
            new_n = df.shape[0]
            # batch.append(new_n)
            n = old_n + new_n
            print(f"current data size {n} (+{new_n})")
            subset_size.append(new_n)
            new_data = X_df[:n]
            U = df.values

            start_date = data_ori['date'].iloc[old_n]
            end_date = data_ori['date'].iloc[n-1]
            print(f"current time period {start_date} ~ {end_date}" )
            print("-------------------------------------------")
            timestamp.append((start_date, end_date))

            deltaV = np.concatenate([H, U], 0)
            shape = deltaV.shape
            deltaH = np.random.rand(rank, shape[1]).astype(np.float64)
            deltaH = np.divide(deltaH, deltaH.max())

            deltaW = np.random.rand(rank + U.shape[0], rank).astype(np.float64)
            deltaW = np.divide(deltaW, deltaW.max())

            ################get the temporal H and W
            cost_last = (LA.norm(deltaV - np.matmul(deltaW, deltaH), ord='fro')) ** 2 / 2
            # print cost_last
            start_time = time.time()

            while epoch < steps and optgap > tolerance:
                WTV = np.matmul(deltaW.T, deltaV)
                WTWH = np.matmul(np.matmul(deltaW.T, deltaW), deltaH)
                deltaH = deltaH * WTV / (WTWH + myeps)
                deltaH[deltaH < myeps] = myeps

                ##########update W
                VHT = np.matmul(deltaV, deltaH.T)
                WHHT = np.matmul(np.matmul(deltaW, deltaH), deltaH.T)
                deltaW = deltaW * VHT / (WHHT + myeps)
                deltaW[deltaW < myeps] = myeps
                epoch += 1
                cost = (LA.norm(deltaV - np.matmul(deltaW, deltaH), ord='fro')) ** 2 / 2
                optgap = abs(cost_last - cost) / cost_last
                cost_last = cost
                # print "This is the epoch %d, the cost is %f" % (epoch,cost)

            W_star = deltaW[0:rank, :]
            W_star2 = deltaW[rank:, :]
            epoch = 0
            optgap = 1000
            ##########initialize

            H_temp = np.random.rand(rank, shape[1]).astype(np.float64)
            H_temp = np.divide(H_temp, H_temp.max())

            Hc = H_temp[0:n_c, :]
            Hd = H_temp[n_c:, :]
            H2 = np.random.rand(rank, shape[1]).astype(np.float64)
            H2 = np.divide(H2, H2.max())

            H2c = H2[0:n_c, :]
            H2d = H2[n_c:, :]

            W2 = np.random.rand(U.shape[0], rank).astype(np.float64)
            W2c = W2[:, 0:n_c]
            W2d = W2[:, n_c:]
            T = np.random.rand(rank, rank).astype(np.float64)
            T = np.divide(T, T.max())

            cost_last = ((LA.norm(H_temp - np.matmul(T, H), ord='fro')) ** 2 / 2 + (
                LA.norm(U - np.matmul(W2, H2), ord='fro')) ** 2 / 2 + alpha_temp * (LA.norm(Hc - H2c, 'fro')) ** 2 +
                         beta * np.sum(np.abs(np.matmul(Hd.T, H2d)))) / (n - old_n)
            # print cost_last
            # cost_display=[]
            # cost_display.append(cost_last)
            while epoch < steps and optgap > tolerance:
                temp = np.matmul(W2d.T, U)
                denom = np.matmul(np.matmul(W2d.T, np.concatenate([W2c, W2d], 1)), np.concatenate([H2c, H2d], 0))
                denom = denom + beta * np.matmul(Hd, np.sign(np.matmul(Hd.T, H2d))) + myeps
                H2d = H2d * temp / (denom + myeps)
                H2d[H2d < myeps] = myeps

                temp = np.matmul(W2c.T, U) + 2 * alpha_temp * Hc
                denom = np.matmul(np.matmul(W2c.T, np.concatenate([W2c, W2d], 1)),
                                  np.concatenate([H2c, H2d], 0)) + 2 * alpha_temp * H2c
                H2c = H2c * temp / (denom + myeps)
                H2c[H2c < myeps] = myeps

                temp = np.matmul(U, H2d.T)
                denom = np.matmul(np.matmul(np.concatenate([W2c, W2d], 1), np.concatenate([H2c, H2d], 0)), H2d.T)
                W2d = W2d * temp / (denom + myeps)
                W2d[W2d < myeps] = myeps

                temp = np.matmul(U, H2c.T)
                denom = np.matmul(np.matmul(np.concatenate([W2c, W2d], 1), np.concatenate([H2c, H2d], 0)), H2c.T)
                W2c = W2c * temp / (denom + myeps)
                W2c[W2c < myeps] = myeps

                temp = np.matmul(T, H)[:n_c, :] + 2 * alpha_temp * H2c
                denom = 2 * alpha_temp * Hc + Hc
                Hc = Hc * temp / (denom + myeps)
                Hc[Hc < myeps] = myeps

                # Hc = Hc + myeps * (Hc < myeps)

                temp = np.matmul(T, H)[n_c:, :]
                denom = Hd + beta * np.matmul(H2d, np.sign(np.matmul(H2d.T, Hd)))
                Hd = Hd * temp / (denom + myeps)
                Hd[Hd < myeps] = myeps
                # Hd = Hd + myeps * (Hd < myeps)

                H_temp = np.concatenate([Hc, Hd], 0)
                temp = np.matmul(H_temp, H.T)
                denom = np.matmul(T, np.matmul(H, H.T))
                T = T * temp / (denom + myeps)
                T[T < myeps] = myeps

                H2 = np.concatenate([H2c, H2d], 0)
                W2 = np.concatenate([W2c, W2d], 1)

                epoch += 1
                cost1 = (LA.norm(H_temp - np.matmul(T, H), ord='fro')) ** 2 / 2
                cost2 = (LA.norm(U - np.matmul(W2, H2), ord='fro')) ** 2 / 2
                cost3 = alpha_temp * (LA.norm(Hc - H2c, ord='fro')) ** 2
                cost4 = beta * np.sum(np.abs(np.matmul(Hd, H2d.T)))
                cost = (cost1 + cost2 + cost3 + cost4) / (n - old_n)
                optgap = abs(cost_last - cost)
                cost_last = cost
                # cost_display.append(cost)
                # print "This is the epoch %d, the cost of WH2 is %f" % (epoch, cost)


            time_list.append(time.time() - start_time)
            # alpha_temp=alpha_temp/np.sqrt(g+1) ###parameter decay, square root decay
            H = deltaH
            W = np.concatenate([np.matmul(W, W_star), W_star2])
            H_list.append(H)
            # W=np.concatenate([np.matmul(W, W1), W2], 0)
            H_U.append(H2)
            W_list.append(W)
            Htemp_list.append(H_temp)
            X_old = new_data

        print(f"Number of timestamps: {len(H_list)}")

        # print(batch)
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
            datatype = 'tweet' if 'on' in str(X_new.loc[n-1,'typeMonth']) else 'news'
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

        # filename.write("PMI: " + ' '.join(map(str, pmi_list)) + '\n')  ##bigger
        ###########calculate common/different scores################
        for i in range(len(H_U)):
            H1 = Htemp_list[i]
            H2 = H_U[i]
            common, differece = com_diff_score(H1, H2, n_c, n_d)
            common_list.append(common)
            differ_list.append(differece)

        print(len(time_list))
        print(len(common_list))
        print(len(differ_list))
        print(len(recon_error))

        filename.write("time: " + ' '.join(map(str, time_list)) + '\n')  # smaller
        filename.write("common: " + ' '.join(map(str, common_list)) + '\n')  # smaller
        filename.write("differ: " + ' '.join(map(str, differ_list)) + '\n')  # ###bigger
        filename.write("error: " + ' '.join(map(str, recon_error)) + '\n')  # ###smaller

    filename.close()


n_topics = range(5, 30, 5)
# steps = range(100, 500, 100)
steps = [100]
# for s in steps:
#     for t in n_topics:
#         c = int(t/2)
#         TDF(t, c, s)


TDF(30, 15, 100)
