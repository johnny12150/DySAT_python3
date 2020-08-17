import networkx as nx
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import LabelEncoder
from ordered_set import OrderedSet
import warnings
warnings.filterwarnings('ignore')


class CategoricalEncoder():
    '''
    This class is for those operating on large data, in which sklearn's LabelEncoder class may take too much time.
    This encoder is only suitable for 1-d array/ list. You may modify it to become n-d compatible.
    '''
    def __init__(self):
        self.f_dict = {}
        self.r_dict = {}
        self.n_elements = 0

    def fit(self, array):
        '''

        :param array: list or np array
        :return: None
        '''

        unique_elements = OrderedSet(array)
        # unique_elements = sorted(unique_elements)
        # print(DUMMY_ITEM in unique_elements)
        # print('-1' in unique_elements)
        self.n_elements = 0
        self.f_dict = {}
        self.r_dict = {}

        for e in unique_elements:
            self.f_dict[e] = self.n_elements
            self.r_dict[self.n_elements] = e
            self.n_elements += 1


    def continue_fit(self, array):
        '''
        Do not refresh n_elements, count from the latest n_elements.
        :param array:
        :return: None
        '''
        unique_elements = set(array)
        for e in unique_elements:
            if e not in self.f_dict:
                self.f_dict[e] = self.n_elements
                self.r_dict[self.n_elements] = e
                self.n_elements += 1


    def reverse_transform(self, transformed_array, to_np=False):
        '''

        :param transformed_array: list or np array
        :return: array: np array with the same shape as input
        '''


        array = [self.r_dict[e] for e in transformed_array]
        if to_np:
            array = np.array(array)
        return array


    def transform(self, array, to_np=False):
        '''

        :param array: array list or np array
        :return: list or np array with the same shape as the input
        '''
        transformed_array = [self.f_dict[e] for e in array]
        if to_np:
            transformed_array = np.array(transformed_array)
        return transformed_array

    def fit_transform(self, array, to_np=False):
        '''

        :param array: array list or np array
        :return: list or np array with the same shape as the input
        '''
        self.fit(array)
        return self.transform(array, to_np)


diginetica = 'F:/data/dataset-train-diginetica/train-item-views.csv'
yoochose = 'F:/data/yoochoose-data/yoochoose-clicks.dat'
df = pd.read_csv(diginetica, delimiter=';')
print(df.isna().sum()/df.shape[0])
df_with_user = df.dropna(subset=['userId'], axis=0)
print(len(df.sessionId.unique()), len(df_with_user.sessionId.unique()))
print(df_with_user.eventdate.min(), df_with_user.eventdate.max())
# graph內包含user跟item node
# edge_list = []
# for i, row in tqdm(df_with_user.iterrows()):
#     edge_list.append((int(row.userId), row.sessionId))
# edge_list = list(set(edge_list))  # 對edge list取set避免重複

# 建成6張的multi-graph, 每張一個月
print(pd.to_datetime(df_with_user.eventdate.astype(str), format='%Y-%m-%d').dt.month.value_counts())  # 每個月各有多少session
df_with_user['month'] = pd.to_datetime(df_with_user.eventdate.astype(str), format='%Y-%m-%d').dt.month

# 過濾太少出現的 user & item
frequent_user = df_with_user['userId'].value_counts()
frequent_user = frequent_user[frequent_user > 2].index.tolist()
frequent_user = df_with_user.loc[df_with_user['userId'].isin(frequent_user)]
frequent_user_idx = frequent_user.index.tolist()
frequent_item = df_with_user['itemId'].value_counts()
frequent_item = frequent_item[frequent_item > 5].index.tolist()
frequent_item = df_with_user.loc[df_with_user['itemId'].isin(frequent_item)]
frequent_item_idx = frequent_item.index.tolist()
intersect_idx = list(set(frequent_item_idx).intersection(set(frequent_user_idx)))
union_idx = list(set(frequent_item_idx + frequent_user_idx))
# filter out useless info, iloc will not work here due to the index order in intersect_idx
# https://stackoverflow.com/questions/19155718/select-pandas-rows-based-on-list-index
df_with_user = df_with_user[df_with_user.index.isin(intersect_idx)]

# 對user跟item id reindex
le = LabelEncoder()
df_with_user['userId'] = le.fit_transform(df_with_user['userId'])
max_userid = le.classes_[-1]
le = LabelEncoder()
df_with_user['itemId'] = le.fit_transform(df_with_user['itemId']) + max_userid

# 依照 T時刻來reindex
ce = CategoricalEncoder()
for j in range(2, 6):
    ce.continue_fit(df_with_user.loc[df_with_user['month'] == j, 'userId'].values)
    df_with_user.loc[df_with_user['month'] == j, 'userId'] = ce.transform(df_with_user.loc[df_with_user['month'] == j, 'userId'].values)
    ce.continue_fit(df_with_user.loc[df_with_user['month'] == j, 'itemId'].values)
    df_with_user.loc[df_with_user['month'] == j, 'itemId'] = ce.transform(df_with_user.loc[df_with_user['month'] == j, 'itemId'].values)

# 整理各月份的edge list
edge_lists = {2: [], 3: [], 4: [], 5: []}
for i, row in tqdm(df_with_user.iterrows()):
    if row.month in edge_lists.keys():
        edge_lists[row.month].append((int(row.userId), int(row.itemId)))

# 依時刻增加edge
graphs = []
for i in edge_lists.keys():
    if i-1 in edge_lists.keys():
        t_edges = set(edge_lists[i] + list(t_edges))
    else:
        t_edges = set(edge_lists[i])
    G = nx.Graph()
    G.add_edges_from(t_edges)
    graphs.append(G)

graphs = np.array(graphs)
np.savez('data/diginetica/graphs.npz', graph=graphs)
