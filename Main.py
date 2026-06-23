import torch
import numpy as np
import sys, copy, math, time, pdb
import pickle as pickle
import scipy.io as sio
import scipy.sparse as ssp
import os.path
import random
import argparse
sys.path.append('%s/../../pytorch_DGCNN' % os.path.dirname(os.path.realpath(__file__)))
from main import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

from util_functions import *
from torch_geometric.data import DataLoader
from model import Net
from DGCNN_embedding import DGCNN
import seaborn as sns
from scipy.sparse import csc_matrix, triu
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.transforms import LineGraph
from torch_geometric.datasets import Planetoid
import scipy.sparse as sp

parser = argparse.ArgumentParser(description='Link Prediction')
# general settings
parser.add_argument('--data-name', default='BUP', help='network name')
parser.add_argument('--train-name', default=None, help='train name')
parser.add_argument('--test-name', default=None, help='test name')
parser.add_argument('--batch-size', type=int, default=50)
parser.add_argument('--max-train-num', type=int, default=5000, 
                    help='set maximum number of train links (to fit into memory)')
parser.add_argument('--max-test-num', type=int, default=10000, 
                    help='set maximum number of test links (to fit into memory)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--test-ratio', type=float, default=0.5,
                    help='ratio of test links')
# model settings
parser.add_argument('--hop', default=2, metavar='S', 
                    help='enclosing subgraph hop number, \
                    options: 1, 2,..., "auto"')
parser.add_argument('--max-nodes-per-hop', default=100, 
                    help='if > 0, upper bound the # nodes per hop by subsampling')
parser.add_argument('--save-model', action='store_true', default=False,
                    help='save the final model')
parser.add_argument('--use_valedges_as_input', action='store_true', default=False,
                    help='save the final model')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print(args)

random.seed(cmd_args.seed)
np.random.seed(cmd_args.seed) 
torch.manual_seed(cmd_args.seed)
if args.hop != 'auto':
    args.hop = int(args.hop)
if args.max_nodes_per_hop is not None:
    args.max_nodes_per_hop = int(args.max_nodes_per_hop)

'''Prepare data'''
args.file_dir = os.path.dirname(os.path.realpath('__file__'))
args.res_dir = os.path.join(args.file_dir, 'results/{}'.format(args.data_name))

if args.train_name is None:
    args.data_dir = os.path.join(args.file_dir, 'data/{}.mat'.format(args.data_name))
    data = sio.loadmat(args.data_dir)
    net = data['net']
    attributes = None
    # check whether net is symmetric (for small nets only)
    if False:
        net_ = net.toarray()
        assert(np.allclose(net_, net_.T, atol=1e-8))
    #Sample train and test links
    train_pos, train_neg, test_pos, test_neg = sample_neg(net, args.test_ratio, max_train_num=args.max_train_num, max_test_num=args.max_test_num)
else:
    args.train_dir = os.path.join(args.file_dir, 'data/{}'.format(args.train_name))
    args.test_dir = os.path.join(args.file_dir, 'data/{}'.format(args.test_name))
    train_idx = np.loadtxt(args.train_dir, dtype=int)
    test_idx = np.loadtxt(args.test_dir, dtype=int)
    max_idx = max(np.max(train_idx), np.max(test_idx))
    net = ssp.csc_matrix((np.ones(len(train_idx)), (train_idx[:, 0], train_idx[:, 1])), shape=(max_idx+1, max_idx+1))
    net[train_idx[:, 1], train_idx[:, 0]] = 1  # add symmetric edges
    net[np.arange(max_idx+1), np.arange(max_idx+1)] = 0  # remove self-loops
    #Sample negative train and test links
    train_pos = (train_idx[:, 0], train_idx[:, 1])
    test_pos = (test_idx[:, 0], test_idx[:, 1])
    train_pos, train_neg, test_pos, test_neg = sample_neg(net, train_pos=train_pos, test_pos=test_pos, max_train_num=args.max_train_num)

    
print(('# train: %d, # test: %d' % (len(train_pos[0]), len(test_pos[0]))))
print(('# train: %d, # test: %d' % (len(train_neg[0]), len(test_neg[0]))))
'''Train and apply classifier'''
A = net.copy()  # the observed network
# A.setdiag(0) 
A[test_pos[0], test_pos[1]] = 0  # mask test links
A[test_pos[1], test_pos[0]] = 0  # mask test links
A.eliminate_zeros()

print("---开始计算模体计数向量---")
thresholds, Graph_motif, edges = compute_motif_similarity_thresholds(A, data_name=args.data_name)
print("---模体计数向量计算结束---\n")



print("---开始提取子图---")
train_graphs, test_graphs, max_n_label = links2subgraphs(A, train_pos, train_neg, test_pos, test_neg, Graph_motif, args.hop, args.max_nodes_per_hop, None)
print("---提取子图结束---\n")

print(('# train: %d, # test: %d' % (len(train_graphs), len(test_graphs))))
print("link2sub_max_n_label", max_n_label)

print("---生成多尺度子图---motif")
train_multiscalegraphs, test_multiscalegraphs = subgraphs2multiscalessubgraphs(train_graphs, test_graphs, thresholds['65%'])
print("---多尺度子图生成结束---\n")

train_graphs_agg = []
for i in train_multiscalegraphs:
    train_graphs_agg.append(i.aggGNNgraph)
test_graphs_agg = []
for i in test_multiscalegraphs:
    test_graphs_agg.append(i.aggGNNgraph)



    
print("--开始生成原始子图对应的线图--")
train_lines = to_linegraphs(train_graphs, max_n_label)      
test_lines = to_linegraphs(test_graphs, max_n_label)
print("--生成原始子图对应的线图结束--\n")
    
print("--开始生成聚合子图对应的线图--")
train_lines_agg = to_linegraphs(train_graphs_agg, max_n_label)
test_lines_agg = to_linegraphs(test_graphs_agg, max_n_label)
print("--生成聚合子图对应的线图结束--\n")

def remove_edge_0_1_from_gnngraph(g):
    if g.num_edges == 0:
        return
    edges = g.edge_pairs.reshape(-1, 2)
    mask = ~(( (edges[:, 0] == 0) & (edges[:, 1] == 1) ) | ( (edges[:, 0] == 1) & (edges[:, 1] == 0) ))
    filtered_edges = edges[mask]
    g.num_edges = filtered_edges.shape[0]
    g.edge_pairs = filtered_edges.flatten()
    # 如果存在边特征，这里同步删除对应索引
    if g.edge_features is not None:
        # 由于双向边特征重复存储，需删除两次
        # 先计算保留的边索引
        keep_idx = np.where(mask)[0]
        new_edge_features = []
        for i in keep_idx:
            new_edge_features.append(g.edge_features[2*i])
            new_edge_features.append(g.edge_features[2*i+1])
        g.edge_features = np.concatenate(new_edge_features, 0)

for g in train_graphs:
    remove_edge_0_1_from_gnngraph(g)
for g in test_graphs:
    remove_edge_0_1_from_gnngraph(g)
for g in train_graphs_agg:
    remove_edge_0_1_from_gnngraph(g)
for g in test_graphs_agg:
    remove_edge_0_1_from_gnngraph(g)
    
print("--开始生成原始子图对应的线图--")
train_graphs = to_datagraphs(train_graphs, max_n_label)
test_graphs = to_datagraphs(test_graphs, max_n_label)
print("--生成原始子图对应的线图结束--\n")

print("--开始生成聚合子图对应的线图--")
train_graphs_agg = to_datagraphs(train_graphs_agg, max_n_label)
test_graphs_agg = to_datagraphs(test_graphs_agg, max_n_label)
print("--生成聚合子图对应的线图结束--\n")


print(len(train_graphs))
print(len(train_lines))

perm = torch.randperm(len(train_graphs))
train_graphs = [train_graphs[i] for i in perm]
train_graphs_agg = [train_graphs_agg[i] for i in perm]
train_lines = [train_lines[i] for i in perm]
train_lines_agg = [train_lines_agg[i] for i in perm]

test_graphs, test_graphs_agg, test_lines, test_lines_agg, val_graphs, val_graphs_agg, val_lines, val_lines_agg =split_val_test_data_ratio(test_graphs, test_graphs_agg, test_lines, test_lines_agg,test_pos, test_neg,test_pos_ratio=0.5,test_neg_ratio=0.5)

# Model configurations

cmd_args.latent_dim = [32, 32, 32]
cmd_args.hidden = 128
cmd_args.out_dim = 0
cmd_args.dropout = True
cmd_args.num_class = 2
cmd_args.mode = 'gpu'
cmd_args.num_epochs = 200
cmd_args.learning_rate = 0.005#0.005
cmd_args.batch_size = 50
cmd_args.printAUC = True
# cmd_args.feat_dim = train_graphs[0].x.shape[1]
cmd_args.feat_dim = (max_n_label + 1)
cmd_args.attr_dim = 0


train_lines_loader = DataLoader(train_lines, batch_size=cmd_args.batch_size, shuffle=False)
test_lines_loader = DataLoader(test_lines, batch_size=cmd_args.batch_size, shuffle=False)
val_lines_loader = DataLoader(val_lines, batch_size=cmd_args.batch_size, shuffle=False)

train_lines_agg_loader = DataLoader(train_lines_agg, batch_size=cmd_args.batch_size, shuffle=False)
test_lines_agg_loader = DataLoader(test_lines_agg, batch_size=cmd_args.batch_size, shuffle=False)
val_lines_agg_loader = DataLoader(val_lines_agg, batch_size=cmd_args.batch_size, shuffle=False)

train_graphs_loader = DataLoader(train_graphs, batch_size=cmd_args.batch_size, shuffle=False)
test_graphs_loader = DataLoader(test_graphs, batch_size=cmd_args.batch_size, shuffle=False)
val_graphs_loader = DataLoader(val_graphs, batch_size=cmd_args.batch_size, shuffle=False)

train_graphs_agg_loader = DataLoader(train_graphs_agg, batch_size=cmd_args.batch_size, shuffle=False)
test_graphs_agg_loader = DataLoader(test_graphs_agg, batch_size=cmd_args.batch_size, shuffle=False)
val_graphs_agg_loader = DataLoader(val_graphs_agg, batch_size=cmd_args.batch_size, shuffle=False)


print("(max_n_label + 1)*2", (max_n_label + 1)*2)

print("cmd_args.feat_dim", cmd_args.feat_dim)



classifier = Net(cmd_args.feat_dim, cmd_args.hidden, cmd_args.latent_dim, cmd_args.dropout)
if cmd_args.mode == 'gpu':
    classifier = classifier.to("cuda")

optimizer = AdamW(classifier.parameters(), lr=cmd_args.learning_rate, weight_decay=5e-3)
# optimizer = optim.Adam(classifier.parameters(), lr=cmd_args.learning_rate)


best_auc = 0
best_aupr = 0
best_auc_val = 0
best_aupr_val = 0
best_auc_acc = 0
best_acc = 0
best_acc_auc = 0
stop_cnt = 0
best_loss = None
best_epoch = None
patience = 20
best_test_loss = None
best_test_epoch = None

for epoch in range(cmd_args.num_epochs):
    classifier.train()
    avg_loss = loop_dataset_gem(classifier, train_graphs_loader, train_graphs_agg_loader, train_lines_loader, train_lines_agg_loader, optimizer=optimizer)
    if not cmd_args.printAUC:
        avg_loss[2] = 0.0
    print(('\033[92maverage training of epoch %d: loss %.5f acc %.5f auc %.5f ap %.5f\033[0m' % (epoch, avg_loss[0], avg_loss[1], avg_loss[2], avg_loss[3])))

    classifier.eval()
    val_loss = loop_dataset_gem(classifier, val_graphs_loader, val_graphs_agg_loader, val_lines_loader, val_lines_agg_loader, None)# optimizer=optimizer)
    val_auc = val_loss[2]
    if not cmd_args.printAUC:
        val_loss[2] = 0.0
        
    if best_auc_val is None:
        best_auc_val = val_loss[2]
    if val_loss[2] >= best_auc_val:
        best_auc_val = val_loss[2]
        
    if best_aupr_val is None:
        best_aupr_val = val_loss[3]
    if val_loss[3] >= best_aupr_val:
        best_aupr_val = val_loss[3]
    print(('\033[93maverage validation of epoch %d: loss %.5f acc %.5f auc %.5f ap %.5f\033[0m' % (epoch, val_loss[0], val_loss[1], val_loss[2], val_loss[3])))
    
    
    test_loss = loop_dataset_gem(classifier, test_graphs_loader, test_graphs_agg_loader, test_lines_loader, test_lines_agg_loader, None)
    if not cmd_args.printAUC:
        test_loss[2] = 0.0
    
    if best_auc is None:
        best_auc = test_loss[2]
    if test_loss[2] >= best_auc:
        best_auc = test_loss[2]
        best_test_epoch = epoch
        
    if best_aupr is None:
        best_aupr = test_loss[3]
    if test_loss[3] >= best_aupr:
        best_aupr = test_loss[3]
        
        
    print(('\033[94maverage test of epoch %d: loss %.5f acc %.5f auc %.5f aupr %.5f\033[0m' % (epoch, test_loss[0], test_loss[1], test_loss[2], test_loss[3])))
    if best_loss is None:
        best_loss = val_loss
    if val_loss[0] <= best_loss[0]:
        best_loss = val_loss
        stop_cnt = 0
    else:
        stop_cnt += 1
    if stop_cnt >= patience:
        break



print('\033[95mFinal test performance: epoch %d: auc %.5f aupr %.5f\033[0m' % (best_test_epoch, best_auc, best_aupr))
print('\033[95mFinal val performance: auc %.5f aupr %.5f\033[0m' % (best_auc_val, best_aupr_val))

with open(f'results_{args.data_name}.txt', 'a') as f:
    f.write(f"{best_auc:.5f} {best_aupr:.5f}\n")

if args.save_model:
    model_name = 'data/{}_model.pth'.format(args.data_name)
    print('Saving final model states to {}...'.format(model_name))
    torch.save(classifier.state_dict(), model_name)
    hyper_name = 'data/{}_hyper.pkl'.format(args.data_name)
    with open(hyper_name, 'wb') as hyperparameters_file:
        pickle.dump(cmd_args, hyperparameters_file)
        print('Saving hyperparameters to {}...'.format(hyper_name))

with open('acc_results.txt', 'a+') as f:
    f.write(str(test_loss[1]) + '\n')

if cmd_args.printAUC:
    with open('auc_results.txt', 'a+') as f:
        f.write(str(test_loss[2]) + '\n')
