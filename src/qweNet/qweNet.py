import sys

import networkx as nx
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from data_utils import betweenness_centrality_parallel as pbc


encoderHaveBatch = True

class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()


# two layer MLP, the first hidden layer, I add a Batchnorm to accelerated the training rate.
class encoder(nn.Module):
    def __init__(self, inDim, numHidden1, outDim):
        super(encoder, self).__init__()
        if encoderHaveBatch == True:
            self.hidden1 = nn.Sequential(nn.Linear(inDim, numHidden1), nn.BatchNorm1d(numHidden1), nn.ReLU(True))
        else:
            self.hidden1 = nn.Sequential(nn.Linear(inDim, numHidden1), nn.ReLU(True))

        self.out = nn.Sequential(nn.Linear(numHidden1, outDim))

    def forward(self, x):
        x = self.hidden1(x)
        x = self.out(x)
        return x



class QweNet(nn.Module):

    def __init__(self, input_dim, latent_dim, T):
        super(QweNet).__init__()
        self.T = T
        self.input_dim = input_dim
        self.latent_dim = latent_dim

    def forward(self, batch_graph_matrix, bath_input):
        # node features
        h = []
        # node neiborhood features
        h_n = []
        h_0 = F.relu(torch.matmul(batch_graph_matrix, nn.Linear(self.input_dim, bias=True)(bath_input)))
        for i in range(self.T):
            if i == 0:
                h.append()


class QweTool:
    def __init__(self):
        self.dataSet = []
        self.train_index = []
        self.val_index = []

    def clearDataset(self):
        self.dataSet.clear()
        self.train_index.clear()
        self.val_index.clear()

    def gen_graph(self, min_size, max_size, graph_type):
        graph_size = np.random.randint(min_size, max_size, 1)
        gen_graphs = [lambda graph_size: nx.erdos_renyi_graph(n=graph_size, p=0.15),
                      lambda graph_size: nx.connected_watts_strogatz_graph(n=graph_size, k=3, p=0.1),
                      lambda graph_size: nx.barabasi_albert_graph(n=graph_size, m=5),
                      lambda graph_size: nx.powerlaw_cluster_graph(n=graph_size, m=5, p=0.05)
                      ]
        return gen_graphs[graph_type](graph_size)

    def prepareValidData(self, n_data, min_size, max_size, types, isTrain=True):
        print('\ngenerating validation graphs...')
        sys.stdout.flush()
        self.clearDataset()
        assert (len(types) == n_data or len(types) == 1)
        for i in tqdm(range(n_data)):
            graph_type = types[0] if len(types) == 1 else types[i]
            g = self.gen_graph(min_size, max_size, graph_type)
            btres = {}
            pbc(g, btres=btres)
            data = {'graph': g, 'bc': btres}
            self.dataSet.append(data)
            if isTrain:
                self.train_index.append(len(self.dataSet) - 1)
            else:
                self.val_index.append(len(self.dataSet) - 1)

    def train(self, max_epoch):
        assert len(self.dataSet) > 0 and len(self.train_index) > 0 and len(self.val_index) > 0
        for iter in range(max_epoch):
            pass
