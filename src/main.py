from qweNet.qweNet import *
from data_util.generate_bc_feature import generate_bc_feature
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.stats import kendalltau as kd
import torch.optim as optim
EMBEDDING_SIZE = 128
REG_HIDDEN = (int)(EMBEDDING_SIZE / 2)
MIN_SIZE = 100
MAX_SIZE = 200
MAX_EPOCH = 10000
N_VALID = 10 # number of validation graphs
N_TRAIN = 100
BATCH_SIZE = 16
LEARNING_RATE = 0.0002
max_bp_iter = 5     # neighbor propagation steps

def main():
    ########## Train
    btl = QweNet(max_bp_iter, 3, EMBEDDING_SIZE, EMBEDDING_SIZE, REG_HIDDEN, 1, 0)
    System = QweTool()
    System.train(btl, optim.Adam(btl.parameters(), lr = LEARNING_RATE), System.pairwise_ranking_loss, 5500)


def viaualize_graph():
    # main()
    G = nx.powerlaw_cluster_graph(n=30, m=5, p=0.15)
    bt = nx.betweenness_centrality(G)
    res_bc_array = np.array(list(bt.values()))
    node_sizes = 300 / res_bc_array.max() * res_bc_array
    node_color = "r"
    nx.draw(G, node_size=node_sizes, node_colornode_shape="o",
            edge_size=1, with_labels=True, node_color=node_color)
    plt.show()


if __name__ == "__main__":
    G = nx.powerlaw_cluster_graph(n=500, m=5, p=0.15)
    time1 = time.time()

    bc = nx.betweenness_centrality(G)
    bc_list = np.array(list(bc.values()))
    order = np.argsort(-bc_list)

    bc_es, hyedges = generate_bc_feature(G, sampler=0, seed= 42)
    list1 = np.array(list(bc_es.values()))
    order1 = np.argsort(-list1)
    tau1, p_value1 = kd(order, order1)

    bc_es_s, hyedge_s = generate_bc_feature(G, sampler=1, seed= 42)
    list2 = np.array(list(bc_es_s.values()))
    order2 = np.argsort(-list2)
    tau2, p_value2 = kd(order, order2)

    bc_es_ss, _ = generate_bc_feature(G, sampler=2, seed= 42)
    list3 = np.array(list(bc_es_ss.values()))
    order3 = np.argsort(-list3)
    tau3, p_value3 = kd(order, order3)

    print("p_values:{} {} {}".format(p_value1, p_value2, p_value3))
    print("tau:{} {} {}".format(tau1, tau2, tau3))
