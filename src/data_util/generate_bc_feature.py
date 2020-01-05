import math
import random
import gc
import itertools
from copy import deepcopy
from multiprocessing import Pool
import networkx as nx
import numpy as np


# here we get a graph in the type of networkx graph type to generate a estimated BC features
# mainly refer to http://people.seas.harvard.edu/~babis/betweenness-centrality-kdd16.pdf
# and "Almost Linear-Time Algorithms for Adaptive Betweenness Centrality using Hypergraph Sketches"
def generate_bc_feature_parallel(G, eps=0.1, normalized=True, sampler=0, seed=None, process=None):
    pass


def generate_degree_feature(G):
    return nx.degree_centrality(G)


def generate_bc_feature_withAstart(G, eps=0.1, normalized=True, seed=None):
    step = 1 if G.is_directed() else 2
    if seed is not None:
        np.random.seed(seed)
    bc_estimation = dict.fromkeys(G, 0.0)
    node_size = G.order()
    hyedge_size = round(math.log(2 * pow(node_size, 3)) / pow(eps, 2))
    sample_size = min(G.order(), hyedge_size)
    id_src = np.random.permutation(G.order())[:sample_size].tolist()
    id_des = np.random.permutation(G.order())[:sample_size].tolist()
    for i in range(sample_size):
        src = list(G.nodes)[id_src[i]]
        des = list(G.nodes)[id_des[i]]
        if src == des:
            continue
        try:
            hyedge = nx.astar_path(G, src, des)
        except nx.NetworkXNoPath:
            continue
        hyedge.remove(src)
        hyedge.remove(des)
        for node in hyedge:
            bc_estimation[node] += step
    if normalized:
        for node in bc_estimation.keys():
            bc_estimation[node] = bc_estimation[node] / sample_size
    return bc_estimation


def generate_bc_feature(G, eps=0.1, normalized=True, sampler=0, seed=None):
    step = 1 if G.is_directed() else 2
    bc_estimation = dict.fromkeys(G, 0.0)
    node_size = G.order()
    sample_size = round(math.log(2 * pow(node_size, 3)) / pow(eps, 2))
    samplers = [0, 1, 2]
    if sampler not in samplers:
        raise Exception("Not supported sampler")

    if seed:
        random.seed(seed)
    hyedges = []
    for i in range(sample_size):
        src = list(G.nodes)[random.randint(0, node_size - 1)]
        des = list(G.nodes)[random.randint(0, node_size - 1)]
        while src == des:
            src = list(G.nodes)[random.randint(0, node_size - 1)]
            des = list(G.nodes)[random.randint(0, node_size - 1)]
        try:
            paths = nx.all_shortest_paths(G, source=src, target=des)
            paths = [list(path) for path in paths]
        except nx.NetworkXNoPath:
            continue

        if sampler == 0:  ## hyedge
            hyedge = list(paths[random.randint(0, len(paths) - 1)])
            hyedge.remove(src)
            hyedge.remove(des)
            for node in hyedge:
                bc_estimation[node] += step
        elif sampler == 1:  ## original hyedge
            hyedge = list(paths[random.randint(0, len(paths) - 1)])
            hyedge.remove(src)
            hyedge.remove(des)
            if len(hyedge) > 0:
                hyedges.append(hyedge)
        elif sampler == 2:  ## yalg
            hyedge = []
            num_path = 0
            for path in paths:
                num_path += 1
                path.remove(src)
                path.remove(des)
                hyedge += path
            for node in hyedge:
                bc_estimation[node] += step / num_path

    if sampler == 1:
        while len(hyedges) > 1:
            count = {}
            max_degree = 0
            max_node = 0
            for hyedge in hyedges:
                for node in hyedge:
                    value = count.get(node, 0)
                    count[node] = value + 1
            for node, degree in count.items():
                if degree >= max_degree:
                    max_degree = degree
                    max_node = node
            bc_estimation[max_node] = len(hyedges)
            # remove
            hyedges = [hyedge for hyedge in hyedges if max_node not in hyedge]
        # for hyedge in hyedges:
        #     for node in hyedge:
        #         bc_estimation[node]  += step

    if normalized:
        for node in bc_estimation.keys():
            bc_estimation[node] = bc_estimation[node] / sample_size
    return bc_estimation, hyedges
