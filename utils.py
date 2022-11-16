import sys
from os import walk
import pandas as pd
from scipy.stats import entropy
from numpy import linalg as LA
from os import walk
pd.set_option('display.max_column', None)
no_top_words = 10  ##number of words to display
myeps = 1e-16


def display_topics(H, feature_names):
    topics = []
    for topic_idx, topic in enumerate(H):
        topics.append([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]])
    return topics


def display_words(H, feature_names, file):
    for topic_idx, topic in enumerate(H):
        file.write("Topic %d:" % (topic_idx+1))

        file.write("\n")
        file.write(" ".join([feature_names[i]
                             for i in topic.argsort()[:-no_top_words - 1:-1]]))
        file.write("\n")
        file.write("\n")


def com_diff_score(H1, H2, n_c, n_d):
    Hc = H1[0:n_c, :]
    Hd = H1[n_c:, :]
    H2c = H2[0:n_c, :]
    H2d = H2[n_c:, :]
    differ_score = 0
    common_score = (LA.norm(Hc.T - H2c.T, ord='fro')) ** 2
    for i in range(n_d):
        for j in range(n_d):
            differ_score += entropy(Hd[i] + myeps, H2d[j] + myeps)
    return common_score / n_c, differ_score / (2 * n_d ** 2)


def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

def findMonth(x):
    return x[:-3]


### ON-OFF-COMBINE ###
def on_off_com(on_file, off_type):  # off_type : 'title', 'content','summary'
    on_path = 'data/preprocessed/'
    off_path = 'data/off_data/crawled/preprocessed/'

    on_data = pd.read_csv(on_path+on_file)
    on_data['typeMonth'] = on_data['yr_month'].apply(lambda x: 'on'+str(x))
    filenames = next(walk(off_path), (None, None, []))[2]  # [] if no file

    off_df = pd.DataFrame()
    for f in filenames:
        off_df = pd.concat([off_df, pd.read_csv(off_path+f)], ignore_index=True)

    # Merge On-Off data
    off_df['typeMonth'] = off_df['yr_month'].apply(lambda x: 'off'+str(x))
    off_df.rename(columns={off_type:'tweet'}, inplace=True)

    onoff_df = pd.concat([on_data, off_df], ignore_index=True)
    onoff_df['tweet'] = onoff_df['tweet'].apply(lambda x: ' '.join(map(str, eval(x))))
    sorter = ['on202001', 'off202001', 'on202002', 'off202002', 'on202003', 'off202003', 'on202004', 'off202004', 'on202005',
              'off202005', 'on202006', 'off202006', 'on202007', 'off202007', 'on202008', 'off202008', 'on202009', 'off202009',
              'on202010', 'off202010', 'on202011', 'off202011', 'on202012', 'off202012',
              'on202101', 'off202101', 'on202102', 'off202102', 'on202103', 'off202103', 'on202104', 'off202104', 'on202105',
              'off202105', 'on202106', 'off202106', 'on202107', 'off202107', 'on202108', 'off202108', 'on202109', 'off202109',
              'on202110', 'off202110', 'on202111', 'off202111', 'on202112', 'off202112']
    onoff_df.typeMonth = onoff_df.typeMonth.astype("category")
    onoff_df.typeMonth.cat.set_categories(sorter, inplace=True)
    onoff_df.dropna(subset=['typeMonth', 'tweet'], inplace=True)
    onoff_df.sort_values(['typeMonth', 'date'], inplace=True, ignore_index=True, ascending=[True, True])
    return onoff_df




def merge_txt(num, step, path):
    # Reading data from file2
    data = ""
    for root, dirs, files in walk("./results/"+path+'/'+str(step)+'/content/', topdown=False):
        for name in files:
            if "topics_" not in name and num in name and name.endswith(".txt"):
                # Reading data from file1
                with open(root+name) as fp:
                    m = name.replace('.txt', '')
                    model = ''.join([i for i in m if not i.isdigit()])
                    data2 = fp.read()
                    data += model
                    data += "\n"
                    data += data2

    with open('results/'+path+'/'+step+'/content/'+'compare'+num, 'w') as fp:
        fp.write(data)


def mergeData(d1, d2, path):
    on_path = path

    df1 = pd.read_csv(on_path+str(d1)+'.csv')
    df2 = pd.read_csv(on_path+str(d2)+'.csv')
    print(len(df1.index), len(df2.index))
    df = pd.concat([df1, df2], ignore_index=True, sort=True)
    print(df)
    df.to_csv('data/preprocessed/'+str(d1)+str(d2)+'.csv', index=False)


# off_path = 'data/off_data/crawled/preprocessed/'
# filenames = next(walk(off_path), (None, None, []))[2]  # [] if no file
# off_df = pd.DataFrame()
# for f in filenames:
#     off_df = pd.concat([off_df, pd.read_csv(off_path+f)], ignore_index=True)
#
# print(len(off_df.index))

def split_stance(on_file='eng_distilbert_clean.csv'):
    on_path = 'data/preprocessed/'
    on_data = pd.read_csv(on_path+on_file)
    print(on_data)
    pro_data = on_data.loc[on_data['vax_label']==1]
    anti_data = on_data.loc[on_data['vax_label']==0]
    pro_data.to_csv(on_path+on_file.replace('.csv', '_pro.csv'))
    anti_data.to_csv(on_path+on_file.replace('.csv', '_anti.csv'))
    sys.exit()

# split_stance()

def find_tweet(keyword, yr_month):
    on_path = 'data/sorted/eng_distilbert_clean.csv'
    df = pd.read_csv(on_path)
    df = df.loc[df['yr_month'] == int(yr_month)]

    tweet_list = df['tweet'].values.tolist()
    date_list = df['date'].values.tolist()
    for i, t in zip(date_list, tweet_list):
        if keyword in t and i > '2020-03-15':
            print(f"{i}) {t}")

find_tweet('trial', '202003')


