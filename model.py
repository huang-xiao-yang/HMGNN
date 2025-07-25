import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pandas as pd
import numpy as np
import pickle
import dgl
import argparse
from dgl.nn import GATConv, GraphConv
from torch_geometric.data import Data
from collections import Counter
import random
import os
import sys
sys.path.append('/home/huangxiaoyang/NIPS24/our_model/new522/data')
# from our_model.new522.toolsv1 import load_data
from toolsv1 import load_data, mask, metapath_fillct, mtp_edge_index, edge_index_masked
from dgl.nn.pytorch import edge_softmax
from dgl import function as fn
from dgl.nn.pytorch import edge_softmax
from dgl._ffi.base import DGLError
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair

SEED = 4
np.random.seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(1)
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

######################## done our ##############################
############## our model ##################
def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif isinstance(m, GATConv):
        nn.init.xavier_normal_(m.lin_l.weight)
        nn.init.xavier_normal_(m.lin_r.weight)
        nn.init.xavier_normal_(m.att)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Parameter):
        nn.init.xavier_normal_(m)

def softmax_one(x, dim=1):
    x = x - x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x)
    return exp_x / (1 + exp_x.sum(dim=dim, keepdim=True))

class ESGtransform(nn.Module):
    def __init__(self, args):
        super(ESGtransform, self).__init__()
        self.args = args
        self.beta_q10 = nn.Parameter(torch.zeros([1]))
        self.beta_q11 = nn.Linear(9, 5, bias=False)
        self.beta_q12 = nn.Linear(9, 5, bias=False)
        self.beta_q20 = nn.Parameter(torch.zeros([1]))
        self.beta_q21 = nn.Linear(9, 5, bias=False)
        self.beta_q22 = nn.Linear(9, 5, bias=False)
        self.w_k = nn.Linear(5, 5, bias=False)
        self.w_v = nn.Linear(5, 5, bias=False)
        self.ln = nn.LayerNorm(5)
        self.apply(init_weights)

    def forward(self, input_s, input_d):
        def transform(x):
            x_ESG = torch.where(x[:, 5:14] <= 0, 1e-10, x[:, 5:14])
            x_esg = torch.where(x[:, 14:] <= 0, 1e-10, x[:, 14:])
            x_features = self.ln(x[:, 0:5])
            x_k = self.w_k(x_features)
            x_v = self.w_v(x_features)
            return x_ESG, x_esg, x_features, x_k, x_v
        
        x_ESG, x_esg, x_features, x_k, x_v = transform(input_s)
        x_q11 = self.beta_q11(x_esg)
        x_q12 = self.beta_q12(torch.mul(x_ESG, x_esg))
        x_q1 = self.beta_q10 + x_q11 + x_q12
        attn_weight1 = softmax_one(torch.mm(x_q1, x_k.T) / math.sqrt(x_q1.size(-1)), dim=1)
        x1 = torch.mm(attn_weight1, x_v)
        
        x_ESG2, x_esg2, x_features2, x_k2, x_v2 = transform(input_d)
        x_q21 = self.beta_q21(x_esg2)
        x_q22 = self.beta_q22(torch.mul(x_ESG2, x_esg2))
        x_q2 = self.beta_q20 + x_q21 + x_q22
        attn_weight2 = softmax_one(torch.mm(x_q2, x_k2.T) / math.sqrt(x_q2.size(-1)), dim=1)
        x2 = torch.mm(attn_weight2, x_v2)
        
        if not self.args.ablation:
            x_s0 = x1 + x_features
            x_d0 = x2 + x_features2
        else:
            x_s0 = x1
            x_d0 = x2
        return x_s0, x_d0

class AggLSTM(nn.Module):
    def __init__(self, args, input_size=5, hidden_size=5):
        super(AggLSTM, self).__init__()
        self.args = args
        self.hidden_size = hidden_size
        self.esglayer = ESGtransform(self.args)
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        torch.autograd.set_detect_anomaly(True)

    def forward(self, xs, xd, metapath):
        s, d = self.esglayer(xs, xd)
        # s 472个节点 69个元路径 3
        ps_vec = F.embedding(torch.tensor(metapath[0], dtype=torch.long).to(device), s)
        ps_vect = ps_vec.view(-1, ps_vec.shape[2], ps_vec.shape[3])
        is_vec = F.embedding(torch.tensor(metapath[2], dtype=torch.long).to(device), s)
        is_vect = is_vec.reshape(-1, is_vec.shape[2], is_vec.shape[3])

        # d
        pd_vec = F.embedding(torch.tensor(metapath[1], dtype=torch.long).to(device), d)
        pd_vect = pd_vec.reshape(-1, pd_vec.shape[2], pd_vec.shape[3])
        id_vec = F.embedding(torch.tensor(metapath[3], dtype=torch.long).to(device), d)
        id_vect = id_vec.reshape(-1, id_vec.shape[2], id_vec.shape[3])

        
        _, (_, out_ps) = self.lstm(ps_vect)
        _, (_, out_is) = self.lstm(is_vect)
        _, (_, out_pd) = self.lstm(pd_vect)
        _, (_, out_is) = self.lstm(id_vect)
        return (out_ps.view(ps_vec.shape[0], ps_vec.shape[1], -1),
                out_is.view(is_vec.shape[0], is_vec.shape[1], -1),
                out_pd.view(pd_vec.shape[0], pd_vec.shape[1], -1),
                out_is.view(id_vec.shape[0], id_vec.shape[1], -1))

class Metapath_attention(nn.Module):
    def __init__(self, args, feature_dim=5, nhead=5):
        super(Metapath_attention, self).__init__()
        self.args = args
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.aggrneigh = AggLSTM(self.args)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, xs, xd, metapath):
        input_ps, input_is, input_pd, input_id = self.aggrneigh(xs, xd, metapath)
        
        o_ps = self.transformer_encoder(input_ps)
        o_is = self.transformer_encoder(input_is)
        o_ps = self.dropout(o_ps)
        o_is = self.dropout(o_is)
        o_ps = torch.sum(o_ps, dim=1)
        o_is = torch.sum(o_is, dim=1)
        ft_s = torch.cat((o_ps, o_is))
        
        o_pd = self.transformer_encoder(input_pd)
        o_id = self.transformer_encoder(input_id)
        o_pd = self.dropout(o_pd)
        o_id = self.dropout(o_id)
        o_pd = torch.sum(o_pd, dim=1)
        o_id = torch.sum(o_id, dim=1)
        ft_d = torch.cat((o_pd, o_id))
        
        return ft_s, ft_d

class HMG(nn.Module):
    def __init__(self, args):
        super(HMG, self).__init__()
        self.args = args
        self.mtlayer = Metapath_attention(self.args)
        self.convs = nn.ModuleList([
            GATConv(args.in_dim, args.out_dim, num_heads=args.head),
            GATConv(args.out_dim, args.out_dim, num_heads=1)
        ])
        # self.bn = nn.BatchNorm1d(num_features=args.out_dim * args.head)
        self.bn = nn.BatchNorm1d(num_features=5)
        # self.loss_w = nn.Parameter(torch.ones([args.out_dim * args.head, args.out_dim * args.head]))
        self.loss_w = nn.Parameter(torch.ones([args.out_dim, args.out_dim]))
        # self.out = nn.Linear(args.out_dim * args.head, 1)
        self.out = nn.Linear(args.out_dim, 1)
        self.dropout = nn.Dropout(0.1)
        self.direct_mask = torch.ones((args.num_node, args.num_node), dtype=torch.bool)
        self.direct_mask[:args.num_directnode, :args.num_directnode] = False
        
    def forward(self, x_s, x_d, s_edge, d_edge, g_s, g_d, metapath, metaid):
        ft_s, ft_d = self.mtlayer(x_s, x_d, metapath)
        
        new_g_s_x = g_s.ndata['x'].clone()
        new_g_s_x[metaid[0], :5] = ft_s
        g_s.ndata['x'] = new_g_s_x

        new_g_d_x = g_d.ndata['x'].clone()
        new_g_d_x[metaid[1], :5] = ft_d
        g_d.ndata['x'] = new_g_d_x

        xs = F.elu(self.convs[0](g_s, g_s.ndata['x'][:, :5]))
        xs = self.bn(xs)
        xs = F.elu(self.convs[1](g_s, xs)).squeeze(2)
        xs = torch.mean(xs[:1139], dim=1)
        
        xd = F.elu(self.convs[0](g_d, g_d.ndata['x'][:, :5]))
        xd = self.bn(xd)
        xd = F.elu(self.convs[1](g_d, xd)).squeeze(2)
        xd = torch.mean(xd[:1139], dim=1)
        
        y_hat = xs + xd
        y_hat = self.out(y_hat)

        out = torch.mm(xs, self.loss_w.T)
        out = torch.mm(out, self.loss_w)
        out = torch.mm(out, xd.T)
        
        out_direct = out + out.T
        out[self.direct_mask] = out_direct[self.direct_mask]
        
        out_matrix = F.softmax(out, dim=-1)
        
        return out_matrix, y_hat
