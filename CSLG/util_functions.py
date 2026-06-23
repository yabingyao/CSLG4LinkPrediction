import numpy as np
import random
from tqdm import tqdm
import os, sys, pdb, math, time
import pickle as cp
#import _pickle as cp  # python3 compatability
import networkx as nx
import argparse
import scipy.io as sio
import scipy.sparse as ssp
from sklearn import metrics
from gensim.models import Word2Vec
import warnings
warnings.simplefilter('ignore', ssp.SparseEfficiencyWarning)
cur_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append('%s/../../pytorch_DGCNN' % cur_dir)
import multiprocessing as mp
import torch
from torch import Tensor
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.transforms import LineGraph
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from torch_geometric.utils import from_networkx
from torch_geometric.utils import coalesce, cumsum, remove_self_loops, scatter
import math
from scipy.sparse import csr_matrix
from torch_geometric.utils import train_test_split_edges, negative_sampling, to_undirected
from torch_sparse import SparseTensor
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import k_hop_subgraph, to_scipy_sparse_matrix
import torch.nn.functional as F
from scipy.sparse import triu
import subprocess
from scipy.sparse.csgraph import shortest_path


class GNNGraph(object):
    def __init__(self, g, label, node_tags=None, motif_vectors=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a numpy array of continuous node features
        '''
        self.num_nodes = len(node_tags)
        self.node_tags = node_tags
        self.label = label
        self.node_features = node_features  # numpy array (node_num * feature_dim)
        self.degs = list(dict(g.degree).values())
        self.nodes = g.nodes
        self.motif_vectors = motif_vectors

        if len(g.edges()) != 0:
            x, y = list(zip(*g.edges()))
            self.num_edges = len(x)        
            self.edge_pairs = np.ndarray(shape=(self.num_edges, 2), dtype=np.int32)
            self.edge_pairs[:, 0] = x
            self.edge_pairs[:, 1] = y
            self.edge_pairs = self.edge_pairs.flatten()
        else:
            self.num_edges = 0
            self.edge_pairs = np.array([])
        
        # see if there are edge features
        self.edge_features = None
        if nx.get_edge_attributes(g, 'features'):  
            # make sure edges have an attribute 'features' (1 * feature_dim numpy array)
            edge_features = nx.get_edge_attributes(g, 'features')
            assert(type(list(edge_features.values())[0]) == np.ndarray) 
            # need to rearrange edge_features using the e2n edge order
            edge_features = {(min(x, y), max(x, y)): z for (x, y), z in list(edge_features.items())}
            keys = sorted(edge_features)
            self.edge_features = []
            for edge in keys:
                self.edge_features.append(edge_features[edge])
                self.edge_features.append(edge_features[edge])  # add reversed edges
            self.edge_features = np.concatenate(self.edge_features, 0)

class MultiScaleGNNGraph(object):
    'modify:def __init__(self, g, multiscales_g, multiscales_g_lable):'

    'def __init__(self, g, multiscales_g):'

    def __init__(self, g, aggG):
        '''
            orginalGNNgraph: orginal GNNgraph
            multiscales: List of graph in different scales
        '''
        self.orginalGNNgraph = g
        self.aggGNNgraph = aggG
        'self.multiscalegraphs = multiscales_g'
        'delete:self.lable_multiscalegraphs = multiscales_g_lable'


class CustomLineGraph(BaseTransform):
    r"""Converts a graph to its corresponding line-graph
    (functional name: :obj:`line_graph`).

    .. math::
        L(\mathcal{G}) &= (\mathcal{V}^{\prime}, \mathcal{E}^{\prime})

        \mathcal{V}^{\prime} &= \mathcal{E}

        \mathcal{E}^{\prime} &= \{ (e_1, e_2) : e_1 \cap e_2 \neq \emptyset \}

    Line-graph node indices are equal to indices in the original graph's
    coalesced :obj:`edge_index`.
    For undirected graphs, the maximum line-graph node index is
    :obj:`(data.edge_index.size(1) // 2) - 1`.

    New node features are given by old edge attributes.
    For undirected graphs, edge attributes for reciprocal edges
    :obj:`(row, col)` and :obj:`(col, row)` get summed together.

    Args:
        force_directed (bool, optional): If set to :obj:`True`, the graph will
            be always treated as a directed graph. (default: :obj:`False`)
    """

    def __init__(self, force_directed: bool = False) -> None:
        self.force_directed = force_directed

    def forward(self, data: Data) -> Data:
        assert data.edge_index is not None
        edge_index, edge_attr = data.edge_index, data.edge_attr
        N = data.num_nodes

        edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes=N)
        row, col = edge_index

        if self.force_directed or data.is_directed():
            i = torch.arange(row.size(0), dtype=torch.long, device=row.device)

            count = scatter(torch.ones_like(row), row, dim=0,
                            dim_size=data.num_nodes, reduce='sum')
            ptr = cumsum(count)

            cols = [i[ptr[col[j]]:ptr[col[j] + 1]] for j in range(col.size(0))]
            rows = [row.new_full((c.numel(),), j) for j, c in enumerate(cols)]

            row, col = torch.cat(rows, dim=0), torch.cat(cols, dim=0)

            data.edge_index = torch.stack([row, col], dim=0)
            data.x = data.edge_attr
            data.num_nodes = edge_index.size(1)

        else:
            # Compute node indices.
            mask = row < col
            row, col = row[mask], col[mask]

            # edge_mapping = {i: (r.item(), c.item()) for i, (r, c) in enumerate(zip(row, col))}
            edge_mapping = torch.tensor([(r.item(), c.item()) for r, c in zip(row, col)], dtype=torch.long)

            data['mapping'] = edge_mapping

            i = torch.arange(row.size(0), dtype=torch.long, device=row.device)

            (row, col), i = coalesce(
                torch.stack([
                    torch.cat([row, col], dim=0),
                    torch.cat([col, row], dim=0)
                ], dim=0),
                torch.cat([i, i], dim=0),
                N,
            )

            # Compute new edge indices according to `i`.
            count = scatter(torch.ones_like(row), row, dim=0,
                            dim_size=data.num_nodes, reduce='sum')
            joints = list(torch.split(i, count.tolist()))

            def generate_grid(x: Tensor) -> Tensor:
                row = x.view(-1, 1).repeat(1, x.numel()).view(-1)
                col = x.repeat(x.numel())
                return torch.stack([row, col], dim=0)

            joints = [generate_grid(joint) for joint in joints]
            joint = torch.cat(joints, dim=1)
            joint, _ = remove_self_loops(joint)
            N = row.size(0) // 2
            joint = coalesce(joint, num_nodes=N)

            if edge_attr is not None:
                data.x = scatter(edge_attr, i, dim=0, dim_size=N, reduce='sum')
            data.edge_index = joint
            data.num_nodes = edge_index.size(1) // 2

        data.edge_attr = None

        return data

        
def sample_neg(net, test_ratio=0.1, train_pos=None, test_pos=None, max_train_num=None, max_test_num=None):
    # get upper triangular matrix
    net_triu = ssp.triu(net, k=1)
    # sample positive links for train/test
    row, col, _ = ssp.find(net_triu)
    # sample positive links if not specified
    if train_pos is None or test_pos is None:
        perm = random.sample(list(range(len(row))), len(row))
        row, col = row[perm], col[perm]
        split = int(math.ceil(len(row) * (1 - test_ratio)))
        train_pos = (row[:split], col[:split])
        test_pos = (row[split:], col[split:])                                                 
    # if max_train_num is set, randomly sample train links
    if max_train_num is not None:
        perm = np.random.permutation(len(train_pos[0]))[:max_train_num]
        train_pos = (train_pos[0][perm], train_pos[1][perm])
#     if max_test_num is not None:
#         perm = np.random.permutation(len(test_pos[0]))[:max_test_num]
#         test_pos = (test_pos[0][perm], test_pos[1][perm])
    # sample negative links for train/test
    train_num, test_num = len(train_pos[0]), len(test_pos[0])
    neg = ([], [])
    n = net.shape[0]
    print('sampling negative links for train and test')
    while len(neg[0]) < train_num + test_num:
        i, j = random.randint(0, n-1), random.randint(0, n-1)
        if i < j and net[i, j] == 0:
            neg[0].append(i)
            neg[1].append(j)
        else:
            continue
    train_neg  = (neg[0][:train_num], neg[1][:train_num])
    test_neg = (neg[0][train_num:], neg[1][train_num:])
    return train_pos, train_neg, test_pos, test_neg

    
def links2subgraphs(A, train_pos, train_neg, test_pos, test_neg, Graph_motif, h=1, max_nodes_per_hop=None, node_information=None):
    # automatically select h from {1, 2}
    if h == 'auto':
        # split train into val_train and val_test
        _, _, val_test_pos, val_test_neg = sample_neg(A, 0.1)
        val_A = A.copy()
        val_A[val_test_pos[0], val_test_pos[1]] = 0
        val_A[val_test_pos[1], val_test_pos[0]] = 0
        val_auc_CN = CN(val_A, val_test_pos, val_test_neg)
        val_auc_AA = AA(val_A, val_test_pos, val_test_neg)
        print('\033[91mValidation AUC of AA is {}, CN is {}\033[0m'.format(val_auc_AA, val_auc_CN))
        if val_auc_AA >= val_auc_CN:
            h = 2
            print('\033[91mChoose h=2\033[0m')
        else:
            h = 1
            print('\033[91mChoose h=1\033[0m')

    # extract enclosing subgraphs
    max_n_label = {'value': 0}
    def helper(A, links, g_label, Graph_motif):
        '''
        g_list = []
        for i, j in tqdm(zip(links[0], links[1])):
            g, n_labels, n_features = subgraph_extraction_labeling((i, j), A, h, max_nodes_per_hop, node_information)
            max_n_label['value'] = max(max(n_labels), max_n_label['value'])
            g_list.append(GNNGraph(g, g_label, n_labels, n_features))
        return g_list
        '''
        # the new parallel extraction code
        start = time.time()
        pool = mp.Pool(mp.cpu_count())
        results = pool.map_async(parallel_worker, [((i, j), A, Graph_motif, h, max_nodes_per_hop, node_information) for i, j in zip(links[0], links[1])])
        remaining = results._number_left
        pbar = tqdm(total=remaining)
        while True:
            pbar.update(remaining - results._number_left)
            if results.ready(): break
            remaining = results._number_left
            time.sleep(1)
        results = results.get()
        pool.close()
        pbar.close()
        g_list = [GNNGraph(g, g_label, n_labels, features, n_features) for g, n_labels, features, n_features in results]
        max_n_label['value'] = max(max([max(n_labels) for _, n_labels, _ , _ in results]), max_n_label['value'])

        end = time.time()
        print("Time eplased for subgraph extraction: {}s".format(end-start))
        return g_list
        

    print('Enclosing subgraph extraction begins...')
    train_graphs = helper(A, train_pos, 1, Graph_motif) + helper(A, train_neg, 0, Graph_motif)
    test_graphs = helper(A, test_pos, 1, Graph_motif) + helper(A, test_neg, 0, Graph_motif)
    print(max_n_label)
    return train_graphs, test_graphs, max_n_label['value']

def parallel_worker(x):
    return subgraph_extraction_labeling(*x)
    
def subgraph_extraction_labeling(ind, A, Graph_motif, h=1, max_nodes_per_hop=None, node_information=None):
    # extract the h-hop enclosing subgraph around link 'ind'
    dist = 0
    nodes = set([ind[0], ind[1]])
    visited = set([ind[0], ind[1]])
    fringe = set([ind[0], ind[1]])
    
    nodes_dist = [0, 0]
    for dist in range(1, h+1):
        fringe = neighbors(fringe, A)
        fringe = fringe - visited
        visited = visited.union(fringe)
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(fringe):
                fringe = random.sample(fringe, max_nodes_per_hop)
        if len(fringe) == 0:
            break
        nodes = nodes.union(fringe)
        nodes_dist += [dist] * len(fringe)
    # move target nodes to top
    nodes.remove(ind[0])
    nodes.remove(ind[1])
    nodes = [ind[0], ind[1]] + list(nodes) 
    subgraph = A[nodes, :][:, nodes]
    # apply node-labeling

    labels = node_label(subgraph)
    # get node features
    features = None
    if Graph_motif is not None:
        features = Graph_motif[nodes]
    node_features = None
    if node_information is not None:
        node_features = node_information[nodes]
    g = nx.from_scipy_sparse_matrix(subgraph)
    if not g.has_edge(0, 1):
        g.add_edge(0, 1)
        
   
    return g, labels.tolist(), features, node_features


def neighbors(fringe, A):
    # find all 1-hop neighbors of nodes in fringe from A
    res = set()
    for node in fringe:
        nei, _, _ = ssp.find(A[:, node])
        nei = set(nei)
        res = res.union(nei)
    return res

def node_label(subgraph):
    # an implementation of the proposed double-radius node labeling (DRNL)
    K = subgraph.shape[0]
    subgraph_wo0 = subgraph[1:, 1:]
    subgraph_wo1 = subgraph[[0]+list(range(2, K)), :][:, [0]+list(range(2, K))]
    dist_to_0 = ssp.csgraph.shortest_path(subgraph_wo0, directed=False, unweighted=True)
    dist_to_0 = dist_to_0[1:, 0]
    dist_to_1 = ssp.csgraph.shortest_path(subgraph_wo1, directed=False, unweighted=True)
    dist_to_1 = dist_to_1[1:, 0]
    d = (dist_to_0 + dist_to_1).astype(int)
    d_over_2, d_mod_2 = np.divmod(d, 2)
    labels = 1 + np.minimum(dist_to_0, dist_to_1).astype(int) + d_over_2 * (d_over_2 + d_mod_2 - 1)
    labels = np.concatenate((np.array([1, 1]), labels))
    labels[np.isinf(labels)] = 0
    labels[labels>1e6] = 0  # set inf labels to 0
    labels[labels<-1e6] = 0  # set -inf labels to 0
    return labels

# def node_label(subgraph):
#     """
#     Double-Radius Node Labeling (DRNL) implementation.
#     Returns node labels for a subgraph where node 0 and node 1 are the target link endpoints.
#     """
#     K = subgraph.shape[0]
#
#     # Remove node 0 and node 1 respectively from subgraph
#     subgraph_wo0 = subgraph[1:, 1:]
#     subgraph_wo1 = subgraph[[0] + list(range(2, K)), :][:, [0] + list(range(2, K))]
#
#     # Compute shortest paths
#     dist_to_0 = ssp.csgraph.shortest_path(subgraph_wo0, directed=False, unweighted=True)[1:, 0]
#     dist_to_1 = ssp.csgraph.shortest_path(subgraph_wo1, directed=False, unweighted=True)[1:, 0]
#
#     # Sum of distances; clean inf before converting to int
#     d = dist_to_0 + dist_to_1
#     d[np.isinf(d)] = 0
#     d = d.astype(int)
#
#     # DRNL labeling formula
#     d_over_2, d_mod_2 = np.divmod(d, 2)
#     labels = 1 + np.minimum(dist_to_0, dist_to_1).astype(int) + d_over_2 * (d_over_2 + d_mod_2 - 1)
#
#     # Add labels for the two target nodes (set to 1 by definition)
#     labels = np.concatenate((np.array([1, 1]), labels))
#
#     # Post-processing: handle invalid values
#     labels[np.isinf(labels)] = 0
#     labels[labels > 1e6] = 0
#     labels[labels < -1e6] = 0
#
#     return labels

def AA(A, test_pos, test_neg):
    # Adamic-Adar score
    A_ = A / np.log(A.sum(axis=1))
    A_[np.isnan(A_)] = 0
    A_[np.isinf(A_)] = 0
    sim = A.dot(A_)
    return CalcAUC(sim, test_pos, test_neg)
    
        
def CN(A, test_pos, test_neg):
    # Common Neighbor score
    sim = A.dot(A)
    return CalcAUC(sim, test_pos, test_neg)


def CalcAUC(sim, test_pos, test_neg):
    pos_scores = np.asarray(sim[test_pos[0], test_pos[1]]).squeeze()
    neg_scores = np.asarray(sim[test_neg[0], test_neg[1]]).squeeze()
    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.hstack([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc

def single_line(batch_graphs):
    pbar = tqdm(batch_graphs, unit='iteration')
    graphs = []
    for graph in pbar:
        #line_graph, labels = to_line(graph, graph.node_tags)
        line_test(graph, graph.node_tags)
        #graphs.append(line_graph)
    return graphs

def gnn_to_line(batch_graph, max_n_label):
    start = time.time()
    pool = mp.Pool(16)
    #pool = mp.Pool(mp.cpu_count())
    results = pool.map_async(parallel_line_worker, [(graph, max_n_label) for graph in batch_graph])
    remaining = results._number_left
    pbar = tqdm(total=remaining)
    while True:
        pbar.update(remaining - results._number_left)
        if results.ready(): break
        remaining = results._number_left
        time.sleep(1)
    results = results.get()
    pool.close()
    pbar.close()
    g_list = [g for g in results]
    return g_list

def parallel_line_worker(x):
    return to_line(*x)

def to_line(graph, max_n_label):
    edges = graph.edge_pairs
    edge_feas = edge_fea(graph, max_n_label)/2
    edges, feas = to_undirect(edges, edge_feas)
    edges = torch.tensor(edges)
    data = Data(edge_index=edges, edge_attr=feas)
    data.num_nodes = graph.num_nodes
    data = LineGraph()(data)
    data.num_nodes = graph.num_edges
    data['y'] = torch.tensor([graph.label])
    return data

def to_edgepairs(graph):
    x, y = zip(*graph.edges())
    num_edges = len(x)
    edge_pairs = np.ndarray(shape=(num_edges, 2), dtype=np.int32)
    edge_pairs[:, 0] = x
    edge_pairs[:, 1] = y
    edge_pairs = edge_pairs.flatten()
    return edge_pairs

def to_datagraphs(batch_graphs, max_n_label):
    graphs = []
    line_graphs = []
    pbar = tqdm(batch_graphs, unit='iteration')
    for graph in pbar:
        edges = graph.edge_pairs

        node_feas = edge_fea(graph, max_n_label)
#         node_feas = torch.tensor(graph.node_features, dtype=torch.float)
        # node_feas = node_fea_motif(graph, file_in, file_out)
        edges = to_undirect_scale(edges)
        edges = torch.tensor(edges)
        data = Data(x=node_feas, edge_index=edges)

        data.num_nodes = graph.num_nodes
        data['y'] = torch.tensor([graph.label])
        
        graphs.append(data)
    return graphs


def to_linegraphs(batch_graphs, max_n_label):
    graphs = []
    pbar = tqdm(batch_graphs, unit='iteration')
    for graph in pbar:
        edges = graph.edge_pairs
        edge_feas = edge_fea(graph, max_n_label) / 2
        edges, feas = to_undirect(edges, edge_feas)

        edges = torch.tensor(edges).long()  
        data = Data(edge_index=edges, edge_attr=feas)
        data.num_nodes = graph.num_nodes

        data = LineGraph()(data)  

        data['y'] = torch.tensor([graph.label])
        graphs.append(data)
    return graphs


def edge_fea(graph, max_n_label):
    # one-hot 标签特征
#     node_tag = torch.zeros(graph.num_nodes, max_n_label + 1)
    node_tag = torch.zeros(int(graph.num_nodes), int(max_n_label) + 1)

    tags = graph.node_tags
    tags = torch.LongTensor(tags).view(-1, 1)
    tags = tags.clamp(0, node_tag.size(1) - 1)
    node_tag.scatter_(1, tags, 1)
    return node_tag

def edge_fea2(labels, edges):
    feas = []
    for i in range(edges.shape[1]):
        fea = [labels[edges[0][i]], labels[edges[1][i]]]
        fea.sort()
        feas.append(fea)
    feas = np.reshape(feas, [-1, 2])
    feas = np.array([feas[:,0], feas[:,1]], dtype=np.float32)
    return torch.tensor(feas/2)
    
def to_undirect2(edges):
    edges = np.reshape(edges, (-1,2 ))
    sr = np.array([edges[:,0], edges[:,1]], dtype=np.int64)
    rs = np.array([edges[:,1], edges[:,0]], dtype=np.int64)
    target_edge = np.array([[0,1],[1,0]])
    return np.concatenate([target_edge, sr, rs], axis=1)

def to_undirect(edges, edge_fea):
    edges = np.reshape(edges, (-1, 2))
    edges = edges[edges[:, 0] != edges[:, 1]]  # ❗️去除自环

    src = edges[:, 0]
    dst = edges[:, 1]

    edge_index = np.stack([src, dst], axis=0)
    fea_s = edge_fea[src]
    fea_r = edge_fea[dst]
    fea_body1 = torch.cat([fea_s, fea_r], dim=1)
#     fea_body1 = fea_s + fea_r
    
    # reverse
    edge_index_rev = np.stack([dst, src], axis=0)
    fea_body2 = torch.cat([fea_r, fea_s], dim=1)
#     fea_body2 = fea_r + fea_s

    final_edge_index = np.concatenate([edge_index, edge_index_rev], axis=1)
    final_edge_attr = torch.cat([fea_body1, fea_body2], dim=0)

    return final_edge_index, final_edge_attr



def to_undirect_scale(edges):
    edges = np.reshape(edges, (-1, 2))
    sr = np.array([edges[:, 0], edges[:, 1]], dtype=np.int64)
    rs = np.array([edges[:, 1], edges[:, 0]], dtype=np.int64)
    return np.concatenate([sr, rs], axis=1)


def line_test(graph, label):
    edges = graph.edge_pairs
    edges= to_undirect2(edges)
    feas = edge_fea2(label, edges)
    data = Data(edge_index=torch.tensor(edges), edge_attr=feas.T)
    data = LineGraph()(data)
    elist = data['edge_index'].numpy()
    #elist = [(elist[0][i], elist[1][i]) for i in range(len(elist[0]))]
    #nx_graph = nx.Graph()
    #nx_graph.add_edges_from(elist)
    #return nx_graph, data['x'].numpy()
    #return nx

    
#新添加的方法
def subgraphs2multiscalessubgraphs(train_graphs, test_graphs, similarity_threshold):
    # extract enclosing subgraphs
    max_n_label = 0

    def helper_multiscale(graphs, max_n_label):
        multiscaleGraphs_list = []
        for g in tqdm(graphs):
            target_node1 = 0
            target_node2 = 1

            numbersubgraphs = 2
            aggG = get_subgraphs(g, target_node1, target_node2, numbersubgraphs, similarity_threshold)
            
#             print(g.num_nodes)
#             print(g.node_features.shape)
#             print(aggG.num_nodes)
#             print(aggG.node_features.shape)
            
            multiscaleGraphs_list.append(MultiScaleGNNGraph(g, aggG))
        return multiscaleGraphs_list

    train_multiscalegraphs = helper_multiscale(train_graphs, max_n_label)
    test_multiscalegraphs = helper_multiscale(test_graphs, max_n_label)

    return train_multiscalegraphs, test_multiscalegraphs
    
def get_subgraphs(G, target_node1, target_node2, numbersubgraphs, similarity_threshold):
    subgraphs = []
    pairs_array = G.edge_pairs.reshape(-1, 2)
    # Convert each row to a tuple and create a list of pairs
    edge_list_temp = [tuple(row) for row in pairs_array]
    graph = nx.Graph(edge_list_temp)
    
    node_features = G.node_features
    # 创建节点属性字典
    attributes = {i: {'vector': G.motif_vectors[i]} for i in range(len(G.motif_vectors))}

    # 设置节点属性
    nx.set_node_attributes(graph, attributes)

    subgraphs.append(graph.copy())

    # Iteratively merge one-hop neighbors of the target node
    for i in range(numbersubgraphs - 1):  # You can adjust the number of iterations as needed
        '''motif'''
        merge_G, node_features = merge_neighbors(graph, target_node1, target_node2, similarity_threshold, node_features)
        
        '''multiscale'''
#         merge_G, labels = merge_neighbors_mlink(graph, target_node1, target_node2)
#         node_features=None
        
        # convert to utils.GNNGraph
        # Create a mapping from node labels to integer indices
        node_indices = {node: idx for idx, node in enumerate(sorted(merge_G.nodes))}
        # Create an empty adjacency matrix
        num_nodes = len(merge_G.nodes)
        adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
        # Populate the adjacency matrix
        for edge in merge_G.edges:
            node1, node2 = edge
            idx1, idx2 = node_indices[node1], node_indices[node2]
            adj_matrix[idx1, idx2] = 1
            adj_matrix[idx2, idx1] = 1  # If the graph is undirected
        net = csr_matrix(adj_matrix)
        # Get the upper triangular part of the sparse matrix
        net_triu = ssp.triu(net, k=1)
        # Get the list of edges from the upper triangular matrix
        edges = list(zip(*net_triu.nonzero()))
       
        g_gnn = nx.Graph(edges)
        adj_matrix = nx.to_numpy_array(g_gnn)
        sparse_adj_matrix = ssp.csc_matrix(adj_matrix)

        node_tags = node_label(sparse_adj_matrix)

        aggG = GNNGraph(g_gnn, G.label, node_tags, None, node_features)
    return aggG

def merge_neighbors(graph, target_node1, target_node2, similarity_threshold, node_features):
    neighbors = list(graph.nodes)
    if len(neighbors) == 0:
        return graph
    label = calculate_labale_base_distance(graph, target_node1, target_node2)

    for neighbor in neighbors:
        if neighbor != target_node1 and neighbor != target_node2:
            for neighbor_of_neighbor in graph.neighbors(neighbor):
                if neighbor_of_neighbor != target_node1 and neighbor_of_neighbor != target_node2 and neighbor_of_neighbor != neighbor:
                    similarity = relative_error_similarity(graph, neighbor, neighbor_of_neighbor)
                    if similarity >= similarity_threshold:
                        for neighbor_of_neighbor2 in graph.neighbors(neighbor):
                            if (neighbor_of_neighbor != neighbor_of_neighbor2):
                                graph.add_edge(neighbor_of_neighbor2, neighbor_of_neighbor)
                        graph.remove_node(neighbor)
                        if node_features is not None:
                            node_features[neighbor] = np.zeros_like(node_features[neighbor])
                        break
    if node_features is not None:
        non_zero_rows = np.any(node_features != 0, axis=1)  # 每一行有非零元素就为True
        # 仅保留那些有非零元素的行
        node_features = node_features[non_zero_rows]
    return graph, node_features

                                   
def calculate_labale_base_distance(graph, target_node1, target_node2):
    distances1 = calculate_distances(graph, target_node1)
    distances2 = calculate_distances(graph, target_node2)
    # label = {node: 1+ min(distances1.get(node, 0),distances2.get(node, 0)) +distances1.get(node, 0) + distances2.get(node, 0) for node in set(distances1) | set(distances2)}
    label = {node: 1 + min(distances1.get(node, 0), distances2.get(node, 0)) +
                   distances1.get(node, 0) + distances2.get(node, 0)
    if distances1.get(node, 0) != float('inf') and distances2.get(node, 0) != float('inf')
    else 0 for node in set(distances1) | set(distances2)}
    label[target_node1] = 1
    label[target_node2] = 1

    return label

def calculate_distances(graph, target_node):
    distances = {}
    for node in graph.nodes:
        if node == target_node:
            distances[node] = 0
        else:
            try:
                distance = nx.shortest_path_length(graph, source=node, target=target_node)
                distances[node] = distance
            except nx.NetworkXNoPath:
                distances[node] = float('inf')  # If there is no path to the target

    return distances

def relative_error_similarity(graph, node1, node2):
    vector1 = graph.nodes[node1]['vector']
    vector2 = graph.nodes[node2]['vector']

    relative_errors = np.abs(vector1 - vector2) / (np.abs(vector1) + np.abs(vector2) + 1e-10)
    similarity = 1 - np.mean(relative_errors)
    return similarity

def loaddataset(name: str, use_valedges_as_input: bool, load=None):
    
    dataset = Planetoid(root="dataset", name=name)
    split_edge = randomsplit(dataset)
    data = dataset[0]
    data.edge_index = to_undirected(split_edge["train"]["edge"].t())
    edge_index = data.edge_index
    data.num_nodes = data.x.shape[0]
    
    data.edge_weight = None 
    print(data.num_nodes, edge_index.max())
    data.adj_t = SparseTensor.from_edge_index(edge_index, sparse_sizes=(data.num_nodes, data.num_nodes))
    data.adj_t = data.adj_t.to_symmetric().coalesce()
    data.max_x = -1
    if name == "ppa":
        data.x = torch.argmax(data.x, dim=-1)
        data.max_x = torch.max(data.x).item()
    elif name == "ddi":
        data.x = torch.arange(data.num_nodes)
        data.max_x = data.num_nodes
    if load is not None:
        data.x = torch.load(load, map_location="cpu")
        data.max_x = -1

    print("dataset split ")
    for key1 in split_edge:
        for key2  in split_edge[key1]:
            print(key1, key2, split_edge[key1][key2].shape[0])


    # Use training + validation edges for inference on test set.
    if use_valedges_as_input:
        val_edge_index = split_edge['valid']['edge'].t()
        full_edge_index = torch.cat([edge_index, val_edge_index], dim=-1)
        data.full_adj_t = SparseTensor.from_edge_index(full_edge_index, sparse_sizes=(data.num_nodes, data.num_nodes)).coalesce()
        data.full_adj_t = data.full_adj_t.to_symmetric()
    else:
        data.full_adj_t = data.adj_t
    return data, split_edge

def randomsplit(dataset, val_ratio: float=0.10, test_ratio: float=0.1):
    def removerepeated(ei):
        ei = to_undirected(ei)
        ei = ei[:, ei[0]<ei[1]]
        return ei
    data = dataset[0]
    data.num_nodes = data.x.shape[0]
    data = train_test_split_edges(data, test_ratio, test_ratio)
    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    num_val = int(data.val_pos_edge_index.shape[1] * val_ratio/test_ratio)
    data.val_pos_edge_index = data.val_pos_edge_index[:, torch.randperm(data.val_pos_edge_index.shape[1])]
    split_edge['train']['edge'] = removerepeated(torch.cat((data.train_pos_edge_index, data.val_pos_edge_index[:, :-num_val]), dim=-1)).t()
    split_edge['valid']['edge'] = removerepeated(data.val_pos_edge_index[:, -num_val:]).t()
    split_edge['valid']['edge_neg'] = removerepeated(data.val_neg_edge_index).t()
    split_edge['test']['edge'] = removerepeated(data.test_pos_edge_index).t()
    split_edge['test']['edge_neg'] = removerepeated(data.test_neg_edge_index).t()
    return split_edge

def seal_processing(dataset, edge_label_index, y):
    data_list = []
    for src, dst in edge_label_index.t().tolist():
        sub_nodes, sub_edge_index, mapping, _ = k_hop_subgraph([src, dst], 1, dataset.edge_index, relabel_nodes=True)
        src, dst = mapping.tolist()
        mask1 = (sub_edge_index[0] != src) | (sub_edge_index[1] != dst)
        mask2 = (sub_edge_index[0] != dst) | (sub_edge_index[1] != src)
        sub_edge_index = sub_edge_index[:, mask1 & mask2]
        src, dst = (dst, src) if src > dst else (src, dst)
        adj = to_scipy_sparse_matrix(sub_edge_index, num_nodes=sub_nodes.size(0)).tocsr()
        idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
        adj_wo_src = adj[idx, :][:, idx]
        idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
        adj_wo_dst = adj[idx, :][:, idx]
        d_src = shortest_path(adj_wo_dst, directed=False, unweighted=True, indices=src)
        d_src = np.insert(d_src, dst, 0, axis=0)
        d_src = torch.from_numpy(d_src)
        d_dst = shortest_path(adj_wo_src, directed=False, 
        unweighted=True, indices=dst-1)
        d_dst = np.insert(d_dst, src, 0, axis=0)
        d_dst = torch.from_numpy(d_dst)
        dist = d_src + d_dst
        half_dist = torch.div(dist, 2, rounding_mode='trunc')
        z = 1 + torch.min(d_src, d_dst) + half_dist * (half_dist + dist % 2 - 1)
#         z = 1 + torch.min(d_src, d_dst) + dist // 2 * (dist // 2 + dist % 2 - 1)
        z[src], z[dst], z[torch.isnan(z)] = 1., 1., 0.
        z = z.to(torch.long)
        node_labels = F.one_hot(z, num_classes=200).to(torch.float)
        node_emb = dataset.x[sub_nodes]
        node_x = torch.cat([node_emb, node_labels], dim=1)
        edge_features = node_x[sub_edge_index[0]] + node_x[sub_edge_index[1]]
        data = Data(x=node_x, z=z, edge_index=sub_edge_index,edge_attr=edge_features, y=y)
        data_list.append(data)
    return data_list


def compute_motif_similarity_thresholds(A, data_name, program_path='./myprogram'):
    print("---开始计算模体计数向量---")

    Graph = nx.from_scipy_sparse_matrix(A)
    dir_path = f"motif_count_vector/{data_name}"
    os.makedirs(dir_path, exist_ok=True)
    print(f"文件夹 {dir_path} 已准备好")

    file_in = f"{dir_path}/{data_name}.in"
    file_out = f"{dir_path}/{data_name}_count.out"
    file_motif = f"{dir_path}/{data_name}_motif.count"

    # 写 .in 文件（节点数 + 边列表）
    valid_edges = sum(1 for u, v in Graph.edges() if u < v)

    with open(file_in, "w") as f:
        f.write(f"{Graph.number_of_nodes()} {valid_edges}\n")
        for u, v in Graph.edges():
            if u < v:
                f.write(f"{u} {v}\n")

    # 调用外部 ORCA/模体计数程序
    command = f"{program_path} node 4 {file_in} {file_out}"
    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result.stdout.decode())
    except subprocess.CalledProcessError as e:
        print(f"执行模体计数程序失败：{e.returncode}")
        print(e.stderr.decode())
        return

    # 将 .out 转为自定义 9维模体向量写入 .count 文件
    with open(file_out, 'r') as fin, open(file_motif, 'w') as fout:
        for line in fin:
            nums = [int(x) for x in line.split()]
            out_vec = [
                nums[0],
                nums[1] + nums[2],
                nums[3],
                nums[4] + nums[5],
                nums[6] + nums[7],
                nums[8],
                nums[9] + nums[10] + nums[11],
                nums[12] + nums[13],
                nums[14]
            ]
            fout.write(' '.join(map(str, out_vec)) + '\n')

    # 加载模体向量
    motif_vectors = np.loadtxt(file_motif, dtype=int)

    # 获取所有上三角边
    upper_triangle = triu(A, k=1)
    rows, cols = upper_triangle.nonzero()
    edges = list(zip(rows, cols))

    # 相似度计算
    def calculate_similarity(vec1, vec2):
        relative_errors = np.abs(vec1 - vec2) / (np.abs(vec1) + np.abs(vec2) + 1e-10)
        return 1 - np.mean(relative_errors)
    
#     def calculate_similarity(vec1, vec2):
#         # 欧氏距离
#         dist = np.linalg.norm(vec1 - vec2, ord=2)
#         # 距离归一化，避免除零
#         norm = np.linalg.norm(vec1, ord=2) + np.linalg.norm(vec2, ord=2) + 1e-10

#         return 1 - dist / norm


    edge_similarities = []
    for u, v in edges:
        sim = calculate_similarity(motif_vectors[u], motif_vectors[v])
        edge_similarities.append(sim)
    
#     print(edge_similarities)
    # 计算相似度阈值
    thresholds = {
        '100%': np.percentile(edge_similarities, 100),
        '95%': np.percentile(edge_similarities, 95),
        '90%': np.percentile(edge_similarities, 90),
        '85%': np.percentile(edge_similarities, 85),
        '80%': np.percentile(edge_similarities, 80),
        '75%': np.percentile(edge_similarities, 75),
        '70%': np.percentile(edge_similarities, 70),
        '65%': np.percentile(edge_similarities, 65),
        '60%': np.percentile(edge_similarities, 60),
        '55%': np.percentile(edge_similarities, 55),
        '50%': np.median(edge_similarities),
        '45%': np.percentile(edge_similarities, 45),
        '40%': np.percentile(edge_similarities, 40),
        '35%': np.percentile(edge_similarities, 35),
        '30%': np.percentile(edge_similarities, 30),
        '25%': np.percentile(edge_similarities, 25),
        '20%': np.percentile(edge_similarities, 20),
        '15%': np.percentile(edge_similarities, 15),
        '10%': np.percentile(edge_similarities, 10),
        '5%': np.percentile(edge_similarities, 5),
        '0%': np.percentile(edge_similarities, 0),
    }

    print("---模体计数向量计算结束---\n")
    for k, v in thresholds.items():
        print(f"相似度阈值 {k}: {v:.4f}")

    return thresholds, motif_vectors, edges

def seal_processing(dataset, edge_label_index, y):
    data_list = []
    for src, dst in edge_label_index.t().tolist():
        sub_nodes, sub_edge_index, mapping, _ = k_hop_subgraph([src, dst], 2, dataset.edge_index, relabel_nodes=True)
        src, dst = mapping.tolist()
        mask1 = (sub_edge_index[0] != src) | (sub_edge_index[1] != dst)
        mask2 = (sub_edge_index[0] != dst) | (sub_edge_index[1] != src)
        sub_edge_index = sub_edge_index[:, mask1 & mask2]
        src, dst = (dst, src) if src > dst else (src, dst)
        adj = to_scipy_sparse_matrix(sub_edge_index, num_nodes=sub_nodes.size(0)).tocsr()
        idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
        adj_wo_src = adj[idx, :][:, idx]
        idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
        adj_wo_dst = adj[idx, :][:, idx]
        d_src = shortest_path(adj_wo_dst, directed=False, unweighted=True, indices=src)
        d_src = np.insert(d_src, dst, 0, axis=0)
        d_src = torch.from_numpy(d_src)
        d_dst = shortest_path(adj_wo_src, directed=False, 
        unweighted=True, indices=dst-1)
        d_dst = np.insert(d_dst, src, 0, axis=0)
        d_dst = torch.from_numpy(d_dst)
        dist = d_src + d_dst
        half_dist = torch.div(dist, 2, rounding_mode='trunc')
        z = 1 + torch.min(d_src, d_dst) + half_dist * (half_dist + dist % 2 - 1)
#         z = 1 + torch.min(d_src, d_dst) + dist // 2 * (dist // 2 + dist % 2 - 1)
        z[src], z[dst], z[torch.isnan(z)] = 1., 1., 0.
        z = z.to(torch.long)
        node_labels = F.one_hot(z, num_classes=200).to(torch.float)
        node_emb = dataset.x[sub_nodes]
        node_x = torch.cat([node_emb, node_labels], dim=1)

        # 5. 在子图层面直接生成边特征（新增部分）
        edge_src, edge_dst = sub_edge_index[0], sub_edge_index[1]
        edge_attr = node_x[edge_src] + node_x[edge_dst]  # 使用相加作为边特征
        
        data = Data(x=node_x, z=z, edge_index=sub_edge_index, y=y, edge_attr=edge_attr)
        data_list.append(data)
    return data_list


def to_tensor_list(t_list):
    return torch.cat([
        torch.tensor([x]) if isinstance(x, (np.ndarray, np.integer)) else (x.unsqueeze(0) if x.dim() == 0 else x)
        for x in t_list
    ], dim=0)

def split_edge_pair(edge_pair, ratio):
    row, col = edge_pair
    assert len(row) == len(col)
    num_edges = len(row)
    split_point = int(num_edges * ratio)

    # 如果是 numpy array，就直接切分
    test = (row[:split_point], col[:split_point])
    val = (row[split_point:], col[split_point:])
    return val, test


def evaluate_with_logistic_regression(train_embeddings, train_targets,
                                      val_embeddings, val_targets,
                                      test_embeddings, test_targets,
                                      solver='lbfgs', max_iter=1000,
                                      penalty='l2', seed=42):
    # 初始化并训练逻辑回归模型
    clf = LogisticRegression(solver=solver,
                             max_iter=max_iter,
                             penalty=penalty,
                             random_state=seed)

    clf.fit(train_embeddings.detach().cpu().numpy(),
            train_targets.detach().cpu().numpy())

    # 获取验证集和测试集的预测概率（正类的概率）
    val_probs = clf.predict_proba(val_embeddings.detach().cpu().numpy())[:, 1]
    test_probs = clf.predict_proba(test_embeddings.detach().cpu().numpy())[:, 1]

    # 转成 numpy 数组作为标签
    val_labels_np = val_targets.detach().cpu().numpy()
    test_labels_np = test_targets.detach().cpu().numpy()

    # 计算 AUC 和 AUPR
    val_auc = roc_auc_score(val_labels_np, val_probs)
    val_aupr = average_precision_score(val_labels_np, val_probs)

    test_auc = roc_auc_score(test_labels_np, test_probs)
    test_aupr = average_precision_score(test_labels_np, test_probs)

    return val_auc, val_aupr, test_auc, test_aupr
