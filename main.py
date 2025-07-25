import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler
import numpy as np
import pickle
import dgl
import argparse
from collections import Counter
import time
import random
import warnings
import os
import sys
sys.path.append('/home/huangxiaoyang/AAAI2026_copy_our') # todo
from toolsv1 import load_data, mask, metapath_fillct, mtp_edge_index, edge_index_masked, adj_matrix
from sklearn.metrics import r2_score
# from modelv2 import HMG, GCN, GCNA
from model import HGB, HAN, MAGNN, HMG, RGCN

os.chdir('/home/huangxiaoyang/AAAI2026_our')
warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser(description='ESG_pre')
parser.add_argument('--gpu_id', type=str, default='0', help="device id to run")
parser.add_argument('--optimizer', type=str, default='adam', help="optimizer")
parser.add_argument('--out_dim', type=str, default=16, help="model's out_dim")
parser.add_argument('--num_node', type=str, default=1139, help="nums of the nodes")
parser.add_argument('--num_directnode', type=str, default=1122, help="nums of the direct nodes")
parser.add_argument('--in_dim', type=int, default=5, help="hidden dim should match with x_featurce dim")
parser.add_argument('--seed', type=int, default=4, help="random seed") # 45 114514 3507 
parser.add_argument('--task', type=str, default='tra', choices=['tra', 'roe', 'npr']) 
parser.add_argument('--lr', type=float, default=4e-7, help="learning rate") # 
parser.add_argument('--model', type=str, default='our', choices=['our', 'rgcn', 'hgb', 'han', 'magnn']) #
parser.add_argument('--iteration', type=int, default=1000, help="iteration") # 
parser.add_argument('--head', type=int, default=5, help="num_heads for layer")
parser.add_argument('--lam', type=float, default=2, help="lam")
parser.add_argument('--ss', type=float, default=1, help="ss")
parser.add_argument('--metapath', type=list, default=[['pi', 'ip'], ['ip', 'pp', 'pi']], help="han_meta_path")
parser.add_argument('--dropout', type=float, default=0.1, help="dropout rate")
parser.add_argument('--ablation', type=bool, default=False, help="ablation") #
args = parser.parse_args()


SEED = args.seed
np.random.seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

if torch.cuda.is_available():

    if args.model == 'our':
        torch.cuda.set_device(0)
    else:
        torch.cuda.set_device(1)
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

if args.ablation:

    path = f'./train/{args.task}/{args.model}_{args.seed}_{args.iteration}_lam{args.lam}_{args.lr}/abla_log.txt'
    dicry = os.path.dirname(path)  
    if not os.path.exists(dicry):
        os.makedirs(dicry)     
    log_file_train = open(path, 'w')
else:
    path = f'./train/{args.task}/{args.model}_{args.seed}_{args.iteration}_lam{args.lam}_{args.lr}/log.txt'
    dicry = os.path.dirname(path)  
    if not os.path.exists(dicry):
        os.makedirs(dicry)  
    log_file_train = open(path, 'w')

writer = SummaryWriter('./boardlog')
suffix = ''

############ DataLoader ############
train_mask, test_mask, test_id, train_id = mask()

if args.model == 'han' or args.model == 'magnn':
    pi, pp, d_edge, s_edge, pip_s, pip_d, ippi_s, ippi_d, mapid= load_data(True)
else:
    pi, pp, d_edge, s_edge, pip_s, pip_d, ippi_s, ippi_d, mapid= load_data(False)

##  feature
with open('data/feature.pickle', 'rb') as file:
    dt_x = pickle.load(file)
## y
if args.task == 'roe':
    with open('data/roe.pickle', 'rb') as file:
        y = pickle.load(file)
if args.task == 'tra': # 
    with open('data/tra.pickle', 'rb') as file:
        y = pickle.load(file)
if args.task == 'npr':
    with open('data/npr.pickle', 'rb') as file:
        y = pickle.load(file)
y = y.unsqueeze(-1)


############# preprocess ###############
##  train_set
pips = metapath_fillct(pip_s, test_id, True)
pipd = metapath_fillct(pip_d, test_id, True)
ippis = metapath_fillct(ippi_s, test_id, False)
ippid = metapath_fillct(ippi_d, test_id, False)
metapath = [pips, pipd, ippis, ippid]

#  train_set
mtp_s, meta_sid = mtp_edge_index(pips, ippis, mapid) # 551个的id，包含了pip和ippi的 # mark
mtp_d, meta_did = mtp_edge_index(pipd, ippid, mapid)
metaid = [meta_sid, meta_did]

# adjancy
adj = adj_matrix(s_edge, True, train_mask).to(device)

# pyg
if args.model == 'our':
    s_edge = torch.cat((s_edge, mtp_s), axis=1)
    s_edge = edge_index_masked(s_edge, test_id).to(device)
    g_s = dgl.graph((s_edge[0], s_edge[1]), num_nodes=dt_x.shape[0]) #
    g_s = dgl.add_self_loop(g_s)
    g_s.ndata['x'] = dt_x

    d_edge = torch.cat((d_edge, mtp_d), axis=1)
    d_edge = edge_index_masked(d_edge, test_id).to(device)
    g_d = dgl.graph((d_edge[0], d_edge[1]), num_nodes=dt_x.shape[0])
    g_d = dgl.add_self_loop(g_d)
    g_d.ndata['x'] = dt_x

if args.model == 'our':
    print('our')
    model = HMG(args).to(device)


################### train ##################
if args.optimizer == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
elif args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=1e-3)
elif args.optimizer == 'Nadam':
    optimizer = torch.optim.NAdam(model.parameters(), lr=args.lr, weight_decay=1e-4)

loss_func = torch.nn.MSELoss().to(device)

def Evaluation(target, pre):
    mae = np.mean(np.abs(target - pre))
    mape = np.mean(np.abs((target - pre) / target))
    rmse = np.sqrt(np.mean(np.power(target - pre, 2)))
    r2 = r2_score(target, pre)
    return mae, mape, rmse, r2

def test_model(adj, model):
    target = y[test_mask].reshape(-1).detach().cpu().numpy()
    with torch.no_grad():
        model.eval()
        if args.model == 'our':
            los_matrix, prediction = model(g_s.ndata['x'], g_d.ndata['x'], s_edge, d_edge, g_s, g_d, metapath, metaid)#model(g_s.x, g_d.x, s_edge, d_edge, g_s, g_d, metapath, metaid)
            sub_direct_matrix = adj[:args.num_directnode, :args.num_directnode] #
            adj[:args.num_directnode, :args.num_directnode] = (sub_direct_matrix + 1) / 2
            likelihood = - torch.mul(los_matrix[test_mask, :][:, test_mask],
                                        adj[test_mask, :][:, test_mask]) + torch.log(
                1 + torch.exp(los_matrix[test_mask, :][:, test_mask]))
            self_loss_2 = torch.mean(likelihood)
            loss = args.ss * torch.sqrt(
                loss_func(prediction[test_mask], y[test_mask]).float()) + self_loss_2 * args.lam

        else:
            if args.model == 'han':
                prediction = model(g=g, h=(g.ndata['feat']['p'], g.ndata['feat']['i']))
            elif args.model == 'hgb':
                prediction = model(g=g, features_list=feat_lst)
            elif args.model == 'magnn':
                prediction = model((g_lists, features_list, type_mask, edge_metapath_indices_lists))
            elif args.model == 'rgcn':
                prediction = model(g, g.ndata['norm'], g.ndata['feat'])
    
            else:
                prediction = model(g_s.x, g_d.x, s_edge, d_edge, g_s, g_d, metapath, metaid)
            loss = torch.sqrt(
                loss_func(prediction[test_mask], y[test_mask]).float())
        pre = prediction[test_mask].reshape(-1).detach().cpu().numpy()
        mae, mape, rmse, r2 = Evaluation(target, pre)
        return mae, mape, rmse, r2, loss

def validate(adj):
    global model
    target = y[train_mask].reshape(-1).detach().cpu().numpy()
    with torch.no_grad():
        model.eval()
        # if args.ss:
        if args.model == 'our':
            los_matrix, prediction = model(g_s.ndata['x'], g_d.ndata['x'], s_edge, d_edge, g_s, g_d, metapath, metaid) #model(g_s.x, g_d.x, s_edge, d_edge, g_s, g_d, metapath, metaid)
            sub_direct_matrix = adj[:args.num_directnode, :args.num_directnode]
            adj[:args.num_directnode, :args.num_directnode] = (sub_direct_matrix + 1) / 2
            likelihood = - torch.mul(los_matrix[train_mask, :][:, train_mask],
                                        adj[train_mask, :][:, train_mask]) + torch.log(
                1 + torch.exp(los_matrix[train_mask, :][:, train_mask]))
            self_loss_2 = torch.mean(likelihood)
            loss = args.ss * torch.sqrt(
                loss_func(prediction[train_mask], y[train_mask]).float()) + self_loss_2 * args.lam
        else:
            if args.model == 'han':
                prediction = model(g=g, h=(g.ndata['feat']['p'], g.ndata['feat']['i']))
            elif args.model == 'hgb':
                prediction = model(g=g, features_list=feat_lst)
            elif args.model == 'magnn':
                prediction = model((g_lists, features_list, type_mask, edge_metapath_indices_lists))
            elif args.model == 'rgcn':
                prediction = model(g, g.ndata['norm'], g.ndata['feat'])

            else:
                prediction = model(g_s.x, g_d.x, s_edge, d_edge, g_s, g_d, metapath, metaid)   
            loss = torch.sqrt(
                loss_func(prediction[train_mask], y[train_mask]).float())
        pre = prediction[train_mask].reshape(-1).detach().cpu().numpy()
        mae, mape, rmse, r2 = Evaluation(target, pre)

        return mae, mape, rmse, r2, loss



def print_write(iteration, mae, mape, rmse, r2, loss):

    print(
        "task:{},epoch:{}  val_loss:{:.4f}  RMSE:{:.4f}  MAE:{:.4f}  MAPE:{:.4f}  r2:{:.4f}".format(args.task, iteration,
                                                                                            loss.item(),
                                                                                            rmse,
                                                                                            mae, mape, r2))
    log_file_train.write(
        "iteration:{}  val_loss:{:.4f}  RMSE:{:.4f}  MAE:{:.4f}  MAPE:{:.4f}  r2:{:.4f}".format(iteration,
                                                                                            loss.item(),
                                                                                            rmse,
                                                                                            mae, mape,
                                                                                            r2) + "\n")
    log_file_train.flush()


def read_path_model(path):
    global model
    if args.model == 'our':
        model = HMG(args)

    else:
        raise SystemExit
    model.load_state_dict(torch.load(path))
    model.to(device)
    return model


def train():
    global optimizer
    global model
    scheduler_lr = lr_scheduler.StepLR(optimizer, 1000, 0.5)

    for iteration in range(args.iteration + 1):
        model.train()
        optimizer.zero_grad()
        if args.model == 'our': 
            los_matrix, prediction = model(g_s.ndata['x'], g_d.ndata['x'], s_edge, d_edge, g_s, g_d, metapath, metaid)#model(g_s.x, g_d.x, s_edge, d_edge, g_s, g_d, metapath, metaid)
            sub_direct_matrix = adj[:args.num_directnode, :args.num_directnode] 
            adj[:args.num_directnode, :args.num_directnode] = (sub_direct_matrix + 1) / 2
            
            likelihood = - torch.mul(los_matrix[train_mask, :][:, train_mask], adj[train_mask, :][:, train_mask]) + torch.log(1 + torch.exp(los_matrix[train_mask, :][:, train_mask]))
            self_loss_2 = torch.mean(likelihood)
            loss = args.ss * torch.sqrt(loss_func(prediction[train_mask], y[train_mask]).float()) + self_loss_2 * args.lam
            train_loss, los_matrix = loss.item(), self_loss_2
            
        else:
            if args.model == 'han':
                prediction = model(g=g, h=(g.ndata['feat']['p'], g.ndata['feat']['i']))
            elif args.model == 'hgb':
                prediction = model(g=g, features_list=feat_lst)
            elif args.model == 'magnn':
                prediction = model((g_lists, features_list, type_mask, edge_metapath_indices_lists))
            elif args.model == 'rgcn':
                prediction = model(g, g.ndata['norm'], g.ndata['feat'])
            else:
                prediction = model(g_s.x, g_d.x, s_edge, d_edge, g_s, g_d, metapath, metaid)
            loss =  torch.sqrt(loss_func(prediction[train_mask], y[train_mask]).float())
        loss.backward()

        optimizer.step()

        scheduler_lr.step()
        
        mae, mape, rmse, r2, loss = validate(adj)

        writer.add_scalars("Node/{}/{}_{}/val/mae/".format(args.task, args.model, str(args.seed)),
                           {"val_mae"+suffix: mae}, iteration)

        writer.add_scalars("Node/{}/{}_{}/val/mape/".format(args.task, args.model, str(args.seed)),
                           {"val_mape"+suffix: mape}, iteration)

        writer.add_scalars("Node/{}/{}_{}/val/rmse/".format(args.task, args.model, str(args.seed)),
                           {"val_rmse"+suffix: rmse}, iteration)

        writer.add_scalars("Node/{}/{}_{}/val/r2/".format(args.task, args.model, str(args.seed)),
                           {"val_r2"+suffix: r2}, iteration)

        if args.model == 'our':
            writer.add_scalars("Node/{}/{}_{}/train&val/los_matrix/".format(args.task, args.model, str(args.seed)),
                               {"train_self_loss"+suffix: los_matrix * args.lam, "train_loss"+suffix: train_loss}, iteration)

        print_write(iteration, mae, mape, rmse, r2, loss)

        
        if args.ablation:
            if not os.path.exists(f'./model/{args.task}/{args.model}_ablation_{args.seed}_{args.lr}'):
                os.makedirs(f'./model/{args.task}/{args.model}_ablation_{args.seed}_{args.lr}')
            torch.save(model.state_dict(), f'./model/{args.task}/{args.model}_ablation_{args.seed}_{args.lr}/model_{iteration}.pth')
        else:
            if not os.path.exists(f'./model/{args.task}/{args.model}_{args.seed}_{args.lr}'):
                os.makedirs(f'./model/{args.task}/{args.model}_{args.seed}_{args.lr}')
            torch.save(model.state_dict(), f'./model/{args.task}/{args.model}_{args.seed}_{args.lr}/model_{iteration}.pth')

num = 0
def test(adj):
    global num
    for iteration in range(args.iteration+1): 
        if args.ablation:
            path = (f'./model/{args.task}/{args.model}_ablation_{args.seed}_{args.lr}/model_{iteration}.pth')
        else:
            path = (f'./model/{args.task}/{args.model}_{args.seed}_{args.lr}/model_{iteration}.pth')
        model = read_path_model(path) 
        mae, mape, rmse, r2, loss = test_model(adj, model) 
        if args.ablation:
            log_file_path = f'./test/{args.task}/{args.model}_{args.seed}_{args.iteration}_lam{args.lam}_{args.lr}/abla_log.txt'
        else:
            log_file_path = f'./test/{args.task}/{args.model}_{args.seed}_{args.iteration}_lam{args.lam}_{args.lr}/log.txt'
        
        log_dir = os.path.dirname(log_file_path) 
        if not os.path.exists(log_dir):
            os.makedirs(log_dir) 
        
        if num == 0:
            with open(log_file_path, 'w') as log_file:
                log_file.write("")
                num += 1
                
        with open(log_file_path, 'a') as log_file: 
            log_message = f"test_loss epoch{iteration}:{loss.item():.4f}  RMSE:{rmse:.4f}  MAE:{mae:.4f}  MAPE:{mape:.4f}  r2:{r2:.4f}\n"
            if iteration == args.iteration:
                t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
                log_note = f'time:{t}  {args.task}  lam:{args.lam}  model:{args.model}  seed:{args.seed}  epoch:{args.iteration}  lr:{args.lr}\ntest_loss:{loss.item():.4f}  RMSE:{rmse:.4f}  MAE:{mae:.4f}  MAPE:{mape:.4f}  r2:{r2:.4f}\n&${rmse:.4f}$&${mae:.4f}$&${mape:.4f}$&${r2:.4f}$\n\n'
                lognote = open ('log_note_our.txt', 'a')
                lognote.write(log_note)
                lognote.flush()
            log_file.write(log_message)
            log_file.flush() 

        print(f"task:{args.task} test_loss:{loss.item():.4f}  RMSE:{rmse:.4f}  MAE:{mae:.4f}  MAPE:{mape:.4f}  r2:{r2:.4f}")



if __name__ == '__main__':
    train()
    test(adj)


