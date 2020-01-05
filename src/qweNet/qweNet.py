import copy
import gc
import pickle
import sys
import time
from multiprocessing.pool import Pool

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import kendalltau as kd
from tensorboardX import SummaryWriter
from torch import nn
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx
from tqdm import tqdm

from data_utils import betweenness_centrality_parallel as pbc
from data_utils import ranktopk, chunks
from generate_bc_feature import generate_bc_feature_withAstart

# use pytorch geometric
# class QweNet(nn.Module):
#     def __init__(self, input_dim, latent_dim, T):
#         super(QweNet).__init__()
#         self.T = T
#         self.input_dim = input_dim
#         self.latent_dim = latent_dim
#
#     def forward(self, batch_graph_matrix, bath_input):
#         # node features
#         h = []
#         # node neiborhood features
#         h_n = []
#         h_0 = F.relu(torch.matmul(batch_graph_matrix, nn.Linear(self.input_dim, bias=True)(bath_input)))
#         for i in range(self.T):
#             if i == 0:
#                 h.append()
EMBEDDING_SIZE = 64
REG_HIDDEN = (int)(EMBEDDING_SIZE / 2)
MIN_SIZE = 4000
MAX_SIZE = 5000
MAX_EPOCH = 10000
N_VALID = 200
N_TRAIN = 2000
BATCH_SIZE = 32
writer = SummaryWriter('./../result/test6')
save_dir = './../model/test6'
INPUT_DIM = 1
num_percheck = 100
T = 4


# class DrBCNet(nn.Module):
#     def __init__(self, node_features_dim, embedding_size=128, T=1):
#         super(DrBCNet, self).__init__()
#         self.embedding_size = embedding_size
#         self.dense1 = nn.Linear(node_features_dim, embedding_size, bias=True)
#         self.dense2 = nn.Linear(embedding_size, embedding_size, bias=True)

class encoder(nn.Module):
    def __init__(self, maxBpIter, nodeFeatDim, embeddingSize, useDropout=True):
        super(encoder, self).__init__()
        self.maxBpIter = maxBpIter
        self.useDropout = useDropout
        self.inputLayer = nn.Sequential(nn.Linear(nodeFeatDim, embeddingSize), nn.BatchNorm1d(embeddingSize),
                                        nn.LeakyReLU())
        # In forward repeat three layers
        self.nodeConv = GCNConv(embeddingSize, embeddingSize)
        # self.nodeConv1 = GraphConv(embeddingSize, embeddingSize)
        # self.nodeConv2 = GraphConv(embeddingSize, embeddingSize)
        self.combine = nn.GRUCell(embeddingSize, embeddingSize)
        self.bn = nn.BatchNorm1d(embeddingSize)

    def forward(self, x, edge_index):
        x = self.inputLayer(x)
        x = x / torch.norm(x, p=2)
        max_x = x
        for i in range(0, self.maxBpIter):
            pre_x = x
            # x1 = self.nodeConv1(x, edge_index)
            # x2 = self.nodeConv2(x, edge_index)
            # x = torch.mean(torch.stack((x1, x2), dim=0), 0)
            neibor = self.nodeConv(x, edge_index)
            # x = self.combine(pre_x, neibor)
            x= neibor
            x = self.bn(x)
            x = F.leaky_relu(x)
            max_x = torch.max(torch.stack((max_x, x), dim=0), 0)[0]
        # x = max_x
        x = x / torch.norm(x, p=2)
        return x


class decoder(nn.Module):
    def __init__(self, decoder_inDim, decoder_numHidden1, decoder_outDim, decoder_auxFeatDim=0, decoderHaveBatch=True,
                 useDropout=True):
        super(decoder, self).__init__()
        self.auxFeatDim = decoder_auxFeatDim
        self.useDropout = useDropout
        if decoderHaveBatch:
            self.hidden1 = nn.Sequential(nn.Linear(decoder_inDim, decoder_numHidden1), nn.BatchNorm1d(
                decoder_numHidden1), nn.LeakyReLU())
        else:
            self.hidden1 = nn.Sequential(nn.Linear(
                decoder_inDim, decoder_numHidden1), nn.LeakyReLU())

        self.out = nn.Sequential(
            nn.Linear(decoder_numHidden1 + decoder_auxFeatDim, decoder_outDim))

    def forward(self, x, aux_feat=None):
        x = self.hidden1(x)
        if self.useDropout:
            x = F.dropout(x, training=self.training)
        if self.auxFeatDim != 0 and aux_feat is not None:
            x = torch.cat((x, aux_feat), 1)
        x = self.out(x)
        return x


class QweNet(nn.Module):
    def __init__(self, encoder_maxBpIter, encoder_nodeFeatDim, encoder_embeddingSize, decoder_inDim, decoder_numHidden1,
                 decoder_outDim, decoder_auxFeatDim=0, decoderHaveBatch=True, useDropout=True):
        super(QweNet, self).__init__()
        self.encoder = encoder(encoder_maxBpIter, encoder_nodeFeatDim, encoder_embeddingSize)
        self.decoder = decoder(decoder_inDim, decoder_numHidden1, decoder_outDim, decoder_auxFeatDim, decoderHaveBatch,
                               useDropout=useDropout)

    def forward(self, data, axu_feat=None):
        # x, edge_index = data.x, data.edge_index
        # x = F.leaky_relu(self.linear0(x))
        # x = x / torch.norm(x, p=2)
        # max_x = x
        # for lv in range(self.T):
        #     x1 = self.conv1_1(x, edge_index)
        #     # x2 = self.conv1_2(x, edge_index)
        #     # x3 = self.conv1_3(x, edge_index)
        #     # x = torch.mean(torch.stack((x1, x2, x3), dim = 0), 0)
        #     x = self.gru(x1, x)
        #     if self.useBN:
        #         x = self.bn1(x)
        #     x = F.leaky_relu(x)
        #     if self.useDropout:
        #         x = F.dropout(x, training=self.training)
        #     # x = self.conv2(x, edge_index)
        #     # if self.useBN:
        #     #     x = self.bn2(x)
        #     # x = F.leaky_relu(x)
        #     # if self.useDropout:
        #     #     x = F.dropout(x, training=self.training)
        #     max_x = torch.max(torch.stack((max_x, x), dim = 0), 0)[0]
        # x = max_x
        # x = x / torch.norm(x, p=2)
        # x = self.decode_layer1(x)
        # x = F.leaky_relu(x)
        # x = self.decode_layer2(x)
        # x = torch.sigmoid(x)
        x, edgeIndex = data.x, data.edge_index
        # aux_feat = []  # add aux_feat's define
        x = self.encoder(x, edgeIndex)
        xlen = x.size()[0]
        # aux_feat = torch.empty(xlen).to(self.device)
        x = self.decoder(x)
        return x


class QweTool:
    def __init__(self):
        self.trainSet = []  # pyg data
        self.testSet = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def build_model(self):
        model = QweNet(encoder_maxBpIter=T, encoder_nodeFeatDim=INPUT_DIM, encoder_embeddingSize=EMBEDDING_SIZE,
                       decoder_inDim=EMBEDDING_SIZE, decoder_numHidden1=REG_HIDDEN, decoder_outDim=1,
                       decoder_auxFeatDim=0, decoderHaveBatch=True, useDropout=True)
        model.apply(self.weights_init)
        return model

    def clearTrainset(self):
        self.trainSet.clear()

    def clearTestset(self):
        self.testSet.clear()

    def gen_graph(self, min_size, max_size, graph_type):
        graph_size = (int)(np.random.randint(min_size, max_size, 1))
        gen_graphs = [lambda graph_size: nx.erdos_renyi_graph(n=graph_size, p=0.15),
                      lambda graph_size: nx.connected_watts_strogatz_graph(n=graph_size, k=3, p=0.1),
                      lambda graph_size: nx.barabasi_albert_graph(n=graph_size, m=5),
                      lambda graph_size: nx.powerlaw_cluster_graph(n=graph_size, m=5, p=0.05)
                      ]
        return gen_graphs[graph_type](graph_size)

    def convert_data(self, g, label=None):
        graph_data = from_networkx(g)
        btres = {}
        if label is None:
            if g.order() > 2000:
                pbc(g, btres=btres)
            else:
                btres = nx.betweenness_centrality(g)
            label = [btres[node] for node in g.nodes]
        graph_data.y = torch.tensor(label, dtype=torch.float32)
        feature = generate_bc_feature_withAstart(g)
        bc_feature = np.array([feature[node] for node in g.nodes]).reshape((g.order(), 1))
        # aux_feature = np.ones((g.order(), 2))
        # node_feature = np.concatenate([bc_feature, aux_feature], axis=1)
        node_feature = bc_feature

        graph_data.x = torch.from_numpy(node_feature).type(torch.float32)
        return graph_data

    def prepareValidData(self, n_data, min_size, max_size, types):
        print('\ngenerating validation graphs...')
        sys.stdout.flush()
        self.clearTestset()
        assert (len(types) == n_data or len(types) == 1)
        for i in tqdm(range(n_data)):
            graph_type = types[0] if len(types) == 1 else types[i]
            g = self.gen_graph(min_size, max_size, graph_type)
            graph_data = self.convert_data(g)
            self.testSet.append(graph_data)

    def gen_new_graph_single(self, para_tuple):
        min_size, max_size, types, graph_index = para_tuple
        graph_type = types[0]
        graph_datas = []
        for _ in graph_index:
            g = self.gen_graph(min_size, max_size, graph_type)
            graph_data = self.convert_data(g)
            graph_datas.append(graph_data)
        return graph_datas

    def prepareValidData_parallel(self, n_data, min_size, max_size, types, processes=None):
        print('\ngenerating validation graphs...')
        assert (len(types) == n_data or len(types) == 1)
        self.clearTestset()
        p = Pool(processes=processes)
        node_divisor = len(p._pool)
        graph_chunks = list(chunks(range(n_data), int(n_data / node_divisor)))
        num_chunks = len(graph_chunks)
        datas = list(p.map(self.gen_new_graph_single,
                           zip([min_size] * num_chunks,
                               [max_size] * num_chunks,
                               [types] * num_chunks,
                               graph_chunks)))
        for data in datas:
            self.testSet += copy.deepcopy(data)
        print("sucessful generate")
        p.close()
        for x in locals().keys():
            del locals()[x]
        gc.collect()
        return

    def gen_new_graph_parallel(self, min_size, max_size, types, num_graph=1000, processes=None):
        print('\ngenerating new training graphs...')
        self.clearTrainset()
        assert (len(types) == num_graph or len(types) == 1)
        p = Pool(processes=processes, maxtasksperchild=40)
        node_divisor = len(p._pool)
        graph_chunks = list(chunks(range(num_graph), int(num_graph / node_divisor)))
        num_chunks = len(graph_chunks)
        datas = list(p.map(self.gen_new_graph_single,
                           zip([min_size] * num_chunks,
                               [max_size] * num_chunks,
                               [types] * num_chunks,
                               graph_chunks)))
        for data in datas:
            self.trainSet += copy.deepcopy(data)
        print("sucessful generate")
        p.close()
        for x in locals().keys():
            del locals()[x]
        gc.collect()
        return

    def gen_new_graph(self, min_size, max_size, types, num_graph=1000):
        print('\ngenerating new training graphs...')
        self.clearTrainset()
        assert (len(types) == num_graph or len(types) == 1)

        for i in tqdm(range(num_graph)):
            graph_type = types[0] if len(types) == 1 else types[i]
            g = self.gen_graph(min_size, max_size, graph_type)
            self.trainSet.append(self.convert_data(g))

    def pairwise_ranking_loss(self, pred, labels, seed=42):
        np.random.seed(seed)
        assert len(pred) == len(labels)
        id_src = np.random.permutation(len(pred))
        id_des = np.random.permutation(len(pred))
        y = torch.tensor(np.ones((len(pred, ))))
        y = y.to(self.device)
        y[labels[id_src] - labels[id_des] < 0] = -1
        src = torch.pow(10, pred[id_src])
        des = torch.pow(10, pred[id_des])
        loss = F.margin_ranking_loss(src, des, y, reduction='sum')
        # pred_res = pred[id_src] - pred[id_des]
        # label_res = labels[id_src] - labels[id_des]
        # loss = F.multilabel_soft_margin_loss(pred_res, torch.sigmoid(label_res), reduction='mean')
        return loss

    def train(self, model, optimizer, criterion, max_epoch):
        model = model.to(self.device)
        types = [0]
        self.prepareValidData_parallel(N_VALID, min_size=MIN_SIZE, max_size=MAX_SIZE, types=types)
        self.gen_new_graph_parallel(MIN_SIZE, MAX_SIZE, types, num_graph=N_TRAIN)
        # self.prepareValidData(N_VALID, min_size=MIN_SIZE, max_size=MAX_SIZE, types=types)
        # self.gen_new_graph(MIN_SIZE, MAX_SIZE, types, num_graph=N_TRAIN)
        vcfile = '%s/ValidValue.csv' % save_dir
        f_out = open(vcfile, 'w')
        for iter in range(max_epoch):
            num = 0
            model.train()
            running_loss = 0.0
            train_set = copy.deepcopy(self.trainSet)
            train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
            for data_batch in train_loader:
                num += data_batch.x.shape[0]
                data_batch = data_batch.to(self.device)
                optimizer.zero_grad()
                true_value = data_batch.y
                with torch.set_grad_enabled(True):
                    pred = model(data_batch)
                    # if flag:
                    #     writer.add_graph(model, data_batch, verbose= True)
                    #     flag = False
                    loss = criterion(pred, true_value)
                    loss.backward()
                    optimizer.step()
                running_loss += loss.item()
            epoch_loss = running_loss
            start = time.clock()
            if iter and iter % 5000 == 0:
                self.gen_new_graph_parallel(MIN_SIZE, MAX_SIZE, types, num_graph=N_TRAIN)
            if iter % num_percheck == 0:
                if iter == 0:
                    N_start = start
                else:
                    N_start = N_end
                frac_topk, frac_kendal = 0.0, 0.0
                test_start = time.time()
                pred = self.predict(model, self.testSet[0])
                writer.add_histogram("%s_pred" % iter, pred)
                for idx in range(N_VALID):
                    run_time, temp_topk, temp_kendal = self.test(model, idx)
                    frac_topk += temp_topk / N_VALID
                    frac_kendal += temp_kendal / N_VALID
                test_end = time.time()
                f_out.write('%.6f, %.6f\n' % (frac_topk, frac_kendal))  # write vc into the file
                f_out.flush()
                print('\niter %d, Top0.01: %.6f, kendal: %.6f' % (iter, frac_topk, frac_kendal))
                print('testing %d graphs time: %.2fs' % (N_VALID, test_end - test_start))
                N_end = time.clock()
                print('%d iterations total time: %.2fs' % (num_percheck, N_end - N_start))
                print('Training loss is %.6f' % epoch_loss)
                sys.stdout.flush()
                model_path = '%s/nrange_iter_%d_%d_%d.pkl' % (save_dir, MIN_SIZE, MAX_SIZE, iter)
                self.saveModel(model_path, model)
                for i, (name, param) in enumerate(model.named_parameters()):
                    if 'bn' not in name:
                        writer.add_histogram(name, param, iter)
                writer.add_scalar('loss', running_loss, iter)
        f_out.close()

    def predict(self, model, data):
        model.eval()
        data = data.to(self.device)
        pred = model(data).cpu().detach().numpy()
        return pred

    def test(self, model, idx):
        data = self.testSet[idx]
        start = time.time()
        pred = self.predict(model, data)
        pred = pred.T.squeeze()
        # np.save('%ietr_pred.npy'% iter, pred)
        end = time.time()
        betw = data.y.cpu().detach().numpy()
        # np.save('%ietr_true.npy' % iter, betw)
        run_time = end - start
        topk = ranktopk(pred, betw, percent=0.01)
        kendal, p_value = kd(betw, pred, nan_policy="omit")
        return run_time, topk, kendal

    def saveModel(self, model_path, model):
        torch.save({
            'model_weights': model.state_dict()
        }, model_path)
        print('model has been saved success!')

    def findModel(self):
        VCFile = './models/ValidValue.csv'
        vc_list = []
        EarlyStop_start = 2
        EarlyStop_length = 1
        num_line = 0
        for line in open(VCFile):
            data = float(line.split(',')[0].strip(','))  # 0:topK; 1:kendal
            vc_list.append(data)
            num_line += 1
            if num_line > EarlyStop_start and data < np.mean(vc_list[-(EarlyStop_length + 1):-1]):
                best_vc = num_line
                break
        best_model_iter = num_percheck * best_vc
        best_model = './models/nrange_iter_%d.ckpt' % (best_model_iter)
        return best_model

    def evaluateSynData(self, data_test, model_file=None):  # test synthetic data
        if model_file is None:  # if user do not specify the model_file
            model_file = self.findModel()
        print('The best model is :%s' % (model_file))
        sys.stdout.flush()
        model = self.build_model(INPUT_DIM)
        self.loadModel(model_file, model)
        frac_run_time, frac_topk, frac_kendal = 0.0, 0.0, 0.0
        self.clearTestset()
        f = open(data_test, 'rb')
        ValidData = pickle.load(f)
        self.testSet = ValidData
        n_test = min(100, len(self.testSet))
        for i in tqdm(range(n_test)):
            run_time, topk, kendal = self.test(model, i)
            frac_run_time += run_time / n_test
            frac_topk += topk / n_test
            frac_kendal += kendal / n_test
        print('\nRun_time, Top0.01, Kendall tau: %.6f, %.6f, %.6f' % (frac_run_time, frac_topk, frac_kendal))
        return frac_run_time, frac_topk, frac_kendal

    def evaluateRealData(self, model_file, graph_file, label_file):  # test real data
        g = nx.read_weighted_edgelist(graph_file)
        sys.stdout.flush()
        model = self.build_model(INPUT_DIM)
        self.loadModel(model_file, model)
        betw_label = []
        for line in open(label_file):
            betw_label.append(float(line.strip().split()[1]))
        start = time.time()
        graph_data = self.convert_data(g, label=betw_label)
        self.testSet.append(graph_data)
        end = time.time()
        run_time = end - start
        start1 = time.time()
        data = self.testSet[0]
        betw_predict = self.predict(model, data)
        end1 = time.time()
        betw_label = data.y
        run_time += end1 - start1
        top001 = ranktopk(betw_label, betw_predict, 0.01)
        top005 = ranktopk(betw_label, betw_predict, 0.05)
        top01 = ranktopk(betw_label, betw_predict, 0.1)
        kendal = kd(betw_label, betw_predict)
        self.clearTestset()
        return top001, top005, top01, kendal, run_time

    def loadModel(self, model_path, model):
        model_hist = torch.load(model_path)
        model.load_state_dict(model_hist['model_weights'])
        print('restore model from file successfully')

    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if 'Batch' not in classname and "Linear" in classname:
            nn.init.kaiming_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.001)
