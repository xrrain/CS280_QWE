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
    btl = QweNet(max_bp_iter, 3, EMBEDDING_SIZE, EMBEDDING_SIZE, REG_HIDDEN, 1, 0, inTraining = False)
    System = QweTool()
    a, b, c = System.evaluateSynData(btl, "/root/CS280_QWE/model/l2norm_iter_200_300_5400.pkl", 1000, 1200)
	

if __name__ == "__main__":
    main()
