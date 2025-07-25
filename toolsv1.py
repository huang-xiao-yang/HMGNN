import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GATConv
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pandas as pd
import numpy as np
import pickle
import dgl
import argparse
import scipy.sparse as sp
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from collections import Counter
import random
import os
from collections import Counter

SEED = 4
np.random.seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
if torch.cuda.is_available():
    # 设置为第二块GPU
    torch.cuda.set_device(1)
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
# done 读取文件
def load_data(ind_reindex):
    # Load data and edge_index
    with open('data/pi.pkl', 'rb') as file:
        # ind_reindex代表i重新编写
        if ind_reindex:
            pi = pickle.load(file)
            pi = pi[pi.year == '2019']

            a = pi.i.unique()
            b = [i for i in range(len(a))]
            k = {i:j for i,j in zip(a, b)}
            
            pi['i']=pi['i'].replace(k)

            pi = pi.values[:, 1:].T
        else:
            pi = pickle.load(file)
            pi = pi[pi.year == '2019']
            pi = pi.values[:, 1:].T

    with open('data/pp.pkl', 'rb') as file:
        pp = pickle.load(file)
        pp = pp[pp.year == '2019']
        pp = pp.values[:, 1:].T

    with open('data/d_id.pkl', 'rb') as file:
        d_edge = pickle.load(file)
        d_edge = torch.tensor(d_edge[d_edge[:,0]=='2019'][:,1:].astype(np.int64).T)
        
    with open('s_id.pkl', 'rb') as file:
        s_edge = pickle.load(file)
        s_edge = torch.tensor(s_edge[s_edge[:,0]=='2019'][:,1:].astype(np.int64).T)
        

    with open('data/pip_s.pkl', 'rb') as file:
        pip_s = pickle.load(file)
        if ind_reindex:
            pip_s['i'] = pip_s['i'].replace(k)
    with open('data/pip_d.pkl', 'rb') as file:
        pip_d = pickle.load(file)

    with open('data/ippi_s.pkl', 'rb') as file:
        ippi_s = pickle.load(file)
        if ind_reindex:
            ippi_s['i_1'] = ippi_s['i_1'].replace(k)
            ippi_s['i_2'] = ippi_s['i_2'].replace(k)
            
        # ippi_s = ippi_s[ippi_s.year == '2019']
    with open('ippi_d.pkl', 'rb') as file:
        ippi_d = pickle.load(file)
        # ippi_d[ippi_d.year == '2019']

    with open('mtid_id.pkl', 'rb') as file:
        mapid = pickle.load(file)    

    return (pi.astype(np.int32), pp.astype(np.int32), d_edge, s_edge, pip_s, pip_d, ippi_s, ippi_d, mapid)

# done 初始化 mask
def mask():
    train_mask = torch.zeros(1139, dtype=bool)
    intervals = [(0, 1121), (1122, 1138)] # 节点1和节点2的起始索引
    # 对每个区间应用独立的 80% True 设置
    for start, end in intervals:
        interval_length = end - start
        num_trues = int(0.8 * interval_length)
        indices = np.random.choice(range(start, end), num_trues, replace=False)
        train_mask[indices] = True
    test_mask = ~train_mask

    train_id = [index for index, value in enumerate(train_mask) if value] # 908
    test_id = [index for index, value in enumerate(train_mask) if not value]
    
    # 有向边和无向边对应的mask（）
    und_mask = torch.ones(train_mask.shape, dtype = torch.bool)
    und_mask[:1122] = False
    return train_mask, test_mask, test_id, train_id


def edge_index_masked(df, test_id):
        del_mask = np.isin(df, test_id).any(axis=0)
        result = df[:,~del_mask]
        return result
def metapath_fillct(df, test_id, pip):
        # 防止信息泄露
        df = df[df.year == '2019'].values[:, 1:]
        del_mask = np.isin(df, test_id).any(axis=1)
        result = df[~del_mask] # 86000删掉验证集后只剩下36732; 随机种子会让元路径数量不一致
        meta_vec = []
        same_length = int(np.percentile([i for i in Counter(result[:, 0].tolist()).values()], 50))  
        if pip:
            for value in np.unique(result[:, 0]):
                    mask = result[:, 0] == value
                    block = result[mask]
                    if block.shape[0] < same_length:
                        padding = torch.tensor([[2278, 2278, 2278]] * (same_length - block.shape[0]))
                        block = np.concatenate((block, padding), axis=0)
                    elif block.shape[0] > same_length:
                        block = block[:same_length]
                    meta_vec.append(block)
        else:
            for value in np.unique(result[:, 0]):
                    mask = result[:, 0] == value
                    block = result[mask]
                    if block.shape[0] < same_length:
                        padding = torch.tensor([[2278, 2278, 2278, 2278]] * (same_length - block.shape[0]))
                        block = np.concatenate((block, padding), axis=0)
                    elif block.shape[0] > same_length:
                        block = block[:same_length]
                    meta_vec.append(block)
        return np.array(meta_vec).astype(np.int64)

# oriid和mtphid，以及mtphid
def mtp_edge_index(pips, ippis, mapid):
    # 返回配好的edge_index， 以及索引
    df = pd.DataFrame(pips[:,0,0].reshape(-1,1), columns=['Numbers'])
    df['Mapped'] = df['Numbers'].map(mapid)
    mtp_s = df
    df = pd.DataFrame(ippis[:,0,0].reshape(-1,1), columns=['Numbers'])
    df['Mapped'] = df['Numbers'].map(mapid)
    mtp_s = pd.concat([mtp_s, df]).values
    return torch.tensor(mtp_s.astype(np.int64).T), torch.tensor(mtp_s[:, 1].astype(np.int64).T)

def graph(d_edge, mtp_d, num_nodes):
    # dgl 输入edge_index
    a = d_edge[0]
    b = d_edge[1]
    c = mtp_d[0]
    d = mtp_d[1]
    
    scr = torch.cat((a.unsqueeze(0), d.unsqueeze(0)), axis=1).view(-1)
    dst = torch.cat((b.unsqueeze(0), c.unsqueeze(0)), axis=1).view(-1)
    
    # pyg
    # torch.cat((d_edge, mtp_d), axis=1)
    return dgl.graph((scr, dst), num_nodes=num_nodes)

# 全图adj
def adj_matrix(s_edge, train_adj, train_mask):
    # 输入原始的edge
    pp = s_edge[:, torch.all(s_edge < 1121, dim=0)] # 0-1121， 1121-1138
    pp_reverse = pp[[1, 0]]
    piip =  s_edge[:, torch.any(s_edge >= 1121, dim=0)] # 有向边索引
    
    # 全图矩阵
    num_nodes = 1139
    adj = torch.zeros(size=(num_nodes, num_nodes), dtype=torch.float32)
    
    # s_edge = torch.Tensor.cpu(s_edge)
    # adj = sp.coo_matrix((torch.ones(s_edge.shape[1]), (s_edge[0, :], s_edge[1, :])),
    #                     shape=(num_nodes, num_nodes),
    #                     dtype=np.complex64).todense()
    
    # 无向图位置更改
    for i in range(piip.shape[1]):
        adj[piip[0, i], piip[1, i]] = 1
        
    
    # 有向图位置进行更改
    for i in range(pp.shape[1]):
        adj[pp[0, i], pp[1, i]] = 1 # i都提到外面去了，也没有对损失函数本质上进行改变
        adj[pp_reverse[0, i], pp_reverse[1, i]] = -1
    
    # # 训练集邻接矩阵 0613:只要在后面用了train_mask，这里就不需要重新运行了
    # if train_adj:
    #     # for id, i in enumerate(train_mask[:1139]):
    #     for id, i in enumerate(train_mask):
    #         if i == False:
    #             adj[id] = 0
    #             adj[:, id] = 0
    
    # # 测试集邻接矩阵         
    # else:
    #     # for id, i in enumerate(train_mask[:1139]):
    #     for id, i in enumerate(train_mask):
    #         if i == True:
    #             adj[id] = 0
    #             adj[:, id] = 0
    
    # mark 有向图是虚部，无向图是实部      
    # adj = torch.tensor(adj)
    # adj_undirect, adj_direct = adj.real, adj.imag
        
    return adj.to(device)


# def adj_matrix(s_edge, train_adj, train_mask):
    # num_nodes = 1139
    # s_edge = torch.Tensor.cpu(s_edge)

    # # 创建空的实部和虚部矩阵
    # real = np.zeros((num_nodes, num_nodes))
    # imag = np.zeros((num_nodes, num_nodes))

    # # 填充实部和虚部
    # for i in range(s_edge.shape[1]):
    #     real[s_edge[0, i], s_edge[1, i]] = 1
    #     imag[s_edge[0, i], s_edge[1, i]] = 1  # 根据你的需要可能这里是0

    # # 对有向图位置进行更改
    # pp = s_edge[:, torch.all(s_edge < 1121, dim=0)]
    # pp_reverse = pp[[1, 0]]
    # for i in range(pp.shape[1]):
    #     imag[pp[0, i], pp[1, i]] = 1
    #     imag[pp_reverse[0, i], pp_reverse[1, i]] = -1

    # # 使用掩码更新训练集或测试集邻接矩阵
    # for id, i in enumerate(train_mask):
    #     if train_adj and not i:
    #         real[id, :] = 0
    #         real[:, id] = 0
    #         imag[id, :] = 0
    #         imag[:, id] = 0
    #     elif not train_adj and i:
    #         real[id, :] = 0
    #         real[:, id] = 0
    #         imag[id, :] = 0
    #         imag[:, id] = 0

    # # 创建复数张量
    # adj = torch.tensor(real + 1j*imag, dtype=torch.complex64)

    # return adj






if __name__ == '__main__':
    data, d_edge, s_edge, pip_s, pip_d, ippi_s, ippi_d, mapid = load_data()
    train_mask, test_mask, test_id, train_id = mask()

    pips = metapath_fillct(pip_s, test_id, True)
    pipd = metapath_fillct(pip_d, test_id, True)
    ippis = metapath_fillct(ippi_s, test_id, False)
    ippid = metapath_fillct(ippi_d, test_id, False)
    mtp_s, mts_id = mtp_edge_index(pips, ippis, mapid)
    mtp_d, mtd_id = mtp_edge_index(pipd, ippid, mapid)
    adj_matrix(s_edge, True, train_mask)
    print(1)
