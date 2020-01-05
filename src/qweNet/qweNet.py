import sys
import time

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
import multiprocessing

from data_util.data_utils import betweenness_centrality_parallel as pbc
from data_util.data_utils import ranktopk
from data_util.generate_bc_feature import generate_bc_feature
import pickle

EMBEDDING_SIZE = 128
REG_HIDDEN = (int)(EMBEDDING_SIZE / 2)
MIN_SIZE = 100
MAX_SIZE = 200
MAX_EPOCH = 10000
N_VALID = 100 # number of validation graphs
N_TRAIN = 1000
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
max_bp_iter = 4     # neighbor propagation steps

writer = SummaryWriter('./../result')

num_percheck = 200

node_feat_dim = 3  # initial node features, [Dc,1,1]
aux_feat_dim = 2   # extra node features in the hidden layer in the decoder, [Dc,CI1,CI2,1]

initialization_stddev = 0.01
combineID = 1   # how to combine self embedding and neighbor embedding,
                   # 0:structure2vec(add node feature and neighbor embedding)
                   #1:graphsage(concatenation); 2:gru


class decoder(nn.Module):
    def __init__(self, decoder_maxBpIter, decoder_nodeFeatDim, decoder_embeddingSize, inTraining=True):
        super(decoder, self).__init__()

        self.maxBpIter = decoder_maxBpIter
        self.inputLayer = nn.Sequential(nn.Linear(decoder_nodeFeatDim, decoder_embeddingSize), nn.LeakyReLU(
            inTraining == False), nn.BatchNorm1d(decoder_embeddingSize))

        #In forward repeat three layers
        self.nodeConv = GCNConv(decoder_embeddingSize, decoder_embeddingSize)
        self.combine = nn.GRUCell(decoder_embeddingSize, decoder_embeddingSize)
        self.outThisCycle = nn.BatchNorm1d(decoder_embeddingSize)

        self.outputLayer = nn.BatchNorm1d(decoder_embeddingSize)

    def forward(self, x, edge_Index):
        x = self.inputLayer(x)
        max_x = x

        for i in range(0, self.maxBpIter):
            pre_x = x
            x = self.nodeConv(x, edge_Index)
            x = self.combine(pre_x, x)
            x = self.outThisCycle(x)

            max_x = torch.max(torch.stack((max_x, x), dim = 0), 0)[0]

        x = max_x
        x = self.outputLayer(x)
        return x


# two layer MLP, the first hidden layer, I add a Batchnorm to accelerated the training rate.
class encoder(nn.Module):
    def __init__(self, encoder_inDim, encoder_numHidden1, encoder_outDim, encoder_auxFeatDim, encoderHaveBatch=True, inTraining=True):
        super(encoder, self).__init__()
        self.auxFeatDim = encoder_auxFeatDim
        if encoderHaveBatch == True:
            self.hidden1 = nn.Sequential(nn.Linear(encoder_inDim, encoder_numHidden1), nn.BatchNorm1d(
                encoder_numHidden1), nn.LeakyReLU(inTraining == False))
        else:
            self.hidden1 = nn.Sequential(nn.Linear(
                encoder_inDim, encoder_numHidden1), nn.LeakyReLU(inTraining == False))

        self.out = nn.Sequential(
            nn.Linear(encoder_numHidden1 + encoder_auxFeatDim, encoder_outDim))

    def forward(self, x, aux_feat):
        x = self.hidden1(x)
        if self.auxFeatDim != 0:
            x = torch.cat((x, aux_feat), 1)
        x = self.out(x)
        return x




class QweNet(nn.Module):

    def __init__(self, decoder_maxBpIter, decoder_nodeFeatDim, decoder_embeddingSize, encoder_inDim, encoder_numHidden1, encoder_outDim, encoder_auxFeatDim, encoderHaveBatch=True, inTraining=True):
        super(QweNet, self).__init__()
        self.decoder = decoder(
            decoder_maxBpIter, decoder_nodeFeatDim, decoder_embeddingSize, inTraining)
        self.encoder = encoder(encoder_inDim, encoder_numHidden1,
                               encoder_outDim, encoder_auxFeatDim, encoderHaveBatch, inTraining)

    def forward(self, data):
        x, edgeIndex = data.x, data.edge_index
        aux_feat = []  # add aux_feat's define
        x = self.decoder(x, edgeIndex)
        xlen = x.size()[0]
        aux_feat = torch.empty(xlen).cuda()
        x = self.encoder(x, aux_feat)
        return x


class QweTool:
    def __init__(self):
        self.trainSet = []  # pyg data
        self.testSet = []
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

    def build_model(self, input_dim):
        return QweNet(node_features_dim=input_dim)

    def clearTrainset(self):
        self.trainSet.clear()
        self.num_train = 0

    def clearTestset(self):
        self.testSet.clear()
        self.num_test = 0

    def gen_graph(self, min_size, max_size, graph_type):
        graph_size = (int)(np.random.randint(min_size, max_size, 1))
        gen_graphs = [lambda graph_size: nx.erdos_renyi_graph(n=graph_size, p=0.15),
                      lambda graph_size: nx.connected_watts_strogatz_graph(
                          n=graph_size, k=8, p=0.1),
                      lambda graph_size: nx.barabasi_albert_graph(
                          n=graph_size, m=4),
                      lambda graph_size: nx.powerlaw_cluster_graph(
                          n=graph_size, m=4, p=0.05)
                      ]
        return gen_graphs[graph_type](graph_size)

    def insert_data(self, g,  isTrain=True, label=None):
        graph_data = from_networkx(g)
        btres = {}
        if label is None:
            if g.order() > 1000:
                pbc(g, btres=btres)
            else:
                btres = nx.betweenness_centrality(g)
            label = [btres[node] for node in g.nodes]
        graph_data.y = torch.tensor(label, dtype=torch.float32)
        feature, _ = generate_bc_feature(g, sampler=2)
        bc_feature = np.array([feature[node]
                               for node in g.nodes]).reshape((g.order(), 1))
        aux_feature = np.ones((g.order(), 2))
        node_feature = np.concatenate([bc_feature, aux_feature], axis=1)

        graph_data.x = torch.from_numpy(node_feature).type(torch.float32)
        if isTrain:
            self.trainSet.append(graph_data)
            self.num_train += 1
        else:
            self.testSet.append(graph_data)
            self.num_test += 1


        

    def prepareValidData(self, n_data, min_size, max_size, types):
        print('\ngenerating validation graphs...')
        sys.stdout.flush()
        self.clearTestset()
        assert (len(types) == n_data or len(types) == 1)
        for i in tqdm(range(n_data)):
            graph_type = types[0] if len(types) == 1 else types[i]
            g = self.gen_graph(min_size, max_size, graph_type)
            self.insert_data(g, isTrain=False)


    def gen_new_graph(self, min_size, max_size, types, num_graph=1000):
        print('\ngenerating new training graphs...')
        self.clearTrainset()
        assert (len(types) == num_graph or len(types) == 1)
        p_list = []
        for i in tqdm(range(num_graph)):
            graph_type = types[0] if len(types) == 1 else types[i]
            g = self.gen_graph(min_size, max_size, graph_type)
            """
            p = multiprocessing.Process(target = self.insert_data, args=(g, True, ))
            p.start()
            p_list.append(p)
            """
            self.insert_data(g, isTrain=True)

    def pairwise_ranking_loss(self, preds, labels, seed=42):

        np.random.seed(seed)
        assert len(preds) == len(labels)
        id_src = np.random.permutation(len(preds))
        id_des = np.random.permutation(len(preds))
        # y = np.ones((len(pred,)))
        # y[labels[id_src] - labels[id_des] < 0] = -1
        # loss = F.margin_ranking_loss(pred[id_src], pred[id_des], y)
        pred_res = preds[id_src] - preds[id_des]
        label_res = labels[id_src] - labels[id_des]

        lossL = nn.BCEWithLogitsLoss(size_average=False)
        label_res = label_res.view(pred_res.size())
        loss = lossL(pred_res, torch.sigmoid(label_res))
        return torch.mean(loss)

    def train(self, model, optimizer, criterion, max_epoch):
        flag = True
        model = model.to(self.device)
        types = [0]
        self.prepareValidData(N_VALID, min_size=MIN_SIZE,
                              max_size=MAX_SIZE, types=types)
        self.gen_new_graph(MIN_SIZE, MAX_SIZE, types, num_graph=N_TRAIN)
        save_dir = './../model'
        vcfile = '%s/ValidValue.csv' % save_dir
        f_out = open(vcfile, 'w')
        for iter in range(max_epoch):
            print(iter)
            num = 0
            model.train()
            running_loss = 0.0
            train_loader = DataLoader(
                self.trainSet, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
            for data_batch in train_loader:
                num += 1
                data_batch = data_batch.to(self.device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    pred = model(data_batch)
                    # if flag:
                    #     writer.add_graph(model, data_batch, verbose= True)
                    #     flag = False
                    true_value = data_batch.y
                    loss = criterion(pred, true_value)
                    loss.backward()
                    optimizer.step()
                running_loss += loss.item()
            start = time.clock()
            if iter and iter % 5000 == 0:
                self.gen_new_graph(MIN_SIZE, MAX_SIZE,
                                   types, num_graph=N_TRAIN)
            if iter % num_percheck == 0:
                if iter == 0:
                    N_start = start
                else:
                    N_start = N_end
                frac_topk, frac_kendal = 0.0, 0.0
                test_start = time.time()
                for idx in range(N_VALID):
                    run_time, temp_topk, temp_kendal = self.test(model, idx)
                    frac_topk += temp_topk / N_VALID
                    frac_kendal += temp_kendal / N_VALID
                test_end = time.time()
                # write vc into the file
                f_out.write('%.6f, %.6f\n' % (frac_topk, frac_kendal))
                f_out.flush()
                print('\niter %d, Top0.01: %.6f, kendal: %.6f' %
                      (iter, frac_topk, frac_kendal))
                print('testing %d graphs time: %.2fs' %
                      (N_VALID, test_end - test_start))
                N_end = time.clock()
                print('%d iterations total time: %.2fs' %
                      (num_percheck, N_end - N_start))
                print('Training loss is %.4f' % loss)
                sys.stdout.flush()
                model_path = '%s/nrange_iter_%d_%d_%d.pkl' % (
                    save_dir, MIN_SIZE, MAX_SIZE, iter)
                self.saveModel(model_path, model)
            for i, (name, param) in enumerate(model.named_parameters()):
                if 'bn' not in name:
                    writer.add_histogram(name, param, 0)
                    writer.add_scalar('loss', running_loss, i)
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
        writer.add_histogram("%s_pred" % iter, pred)
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

    def evaluateSynData(self, model, data_test, model_file=None):  # test synthetic data
        if model_file is None:  # if user do not specify the model_file
            model_file = self.findModel()
        print('The best model is :%s' % (model_file))
        sys.stdout.flush()
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
        print('\nRun_time, Top0.01, Kendall tau: %.6f, %.6f, %.6f' %
              (frac_run_time, frac_topk, frac_kendal))
        return frac_run_time, frac_topk, frac_kendal

    def evaluateRealData(self, model, model_file, graph_file, label_file):  # test real data
        g = nx.read_weighted_edgelist(graph_file, nodetype = int)
        sys.stdout.flush()
        model = model.to(self.device)
        self.loadModel(model_file, model)
        betw_label = []
        for line in open(label_file):
            betw_label.append(float(line.strip().split()[1]))
        start = time.time()
        self.insert_data(g, isTrain=False, label=betw_label)
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
