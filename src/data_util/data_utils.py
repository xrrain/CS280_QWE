import gc
import itertools
from copy import deepcopy
from multiprocessing import Pool

import networkx as nx
import numpy as np


def chunks(l, n):
    """Divide a list of nodes `l` in `n` chunks"""
    l_c = iter(l)
    while 1:
        x = tuple(itertools.islice(l_c, n))
        if not x:
            return
        yield x


def _betmap(G_normalized_weight_sources_tuple):
    """Pool for multiprocess only accepts functions with one argument.
    This function uses a tuple as its only argument. We use a named tuple for
    python 3 compatibility, and then unpack it when we send it to
    `betweenness_centrality_source`
    """
    res = nx.betweenness_centrality_source(*G_normalized_weight_sources_tuple)
    return res


def betweenness_centrality_parallel(G, btres, processes=None):
    """Parallel betweenness centrality  function"""
    p = Pool(processes=processes, maxtasksperchild=40)
    node_divisor = len(p._pool)
    node_chunks = list(chunks(G.nodes(), int(G.order() / node_divisor)))
    num_chunks = len(node_chunks)
    res = p.imap(_betmap,
                 zip([G] * num_chunks,
                     [True] * num_chunks,
                     [None] * num_chunks,
                     node_chunks))

    bt_array = np.zeros((G.order()))
    bt_sc = []
    for pool_res in res:
        bt_sc.append(deepcopy(pool_res))
    for bt in bt_sc:
        res_list = deepcopy(list(bt.values()))
        bt_array += np.array(res_list)
    btres.update(dict(zip(bt_sc[0].keys(), deepcopy(bt_array.tolist()))))
    # delete locals memory to avid memory
    for x in locals().keys():
        del locals()[x]
    gc.collect()
    return
