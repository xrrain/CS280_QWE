import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch.optim as optim

from qweNet.qweNet import QweTool

INPUT_DIM = 1
MAX_EPOCH = 10000
# def main():
#     ########## Train
#     btl = qweNet()
#     btl.Train()


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
    # G = nx.powerlaw_cluster_graph(n=500, m=5, p=0.15)
    # time1 = time.time()
    #
    # bc = nx.betweenness_centrality(G)
    # bc_list = np.array(list(bc.values()))
    # order = np.argsort(-bc_list)
    #
    # bc_es, hyedges = generate_bc_feature(G, sampler=0, seed= 42)
    # list1 = np.array(list(bc_es.values()))
    # order1 = np.argsort(-list1)
    # tau1, p_value1 = kd(order, order1)
    #
    # bc_es_s, hyedge_s = generate_bc_feature(G, sampler=1, seed= 42)
    # list2 = np.array(list(bc_es_s.values()))
    # order2 = np.argsort(-list2)
    # tau2, p_value2 = kd(order, order2)
    #
    # bc_es_ss, _ = generate_bc_feature(G, sampler=2, seed= 42)
    # list3 = np.array(list(bc_es_ss.values()))
    # order3 = np.argsort(-list3)
    # tau3, p_value3 = kd(order, order3)
    #
    # print("p_values:{} {} {}".format(p_value1, p_value2, p_value3))
    # print("tau:{} {} {}".format(tau1, tau2, tau3))
    qwetool = QweTool()
    model = qwetool.build_model()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.1, 0.99))
    qwetool.train(model, optimizer=optimizer, criterion=qwetool.pairwise_ranking_loss, max_epoch=MAX_EPOCH)
