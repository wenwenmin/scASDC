from __future__ import print_function, division
import argparse
import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.nn import Linear
from utils import load_data, load_graph
from GNN import GNNLayer
from evaluation import eva
from preprocess import prepro, normalize_1
import scanpy as sc
from layers import ZINBLoss, MeanAct, DispAct
from calcu_graph import construct_graph
import warnings
import os
import umap
import matplotlib.pyplot as plt

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

torch.cuda.set_device(0)


def dot_product_decode(Z):
    A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
    return A_pred


class AE(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()
        self.enc_1 = Linear(n_input, n_enc_1)
        self.BN1 = nn.BatchNorm1d(n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.BN2 = nn.BatchNorm1d(n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.BN3 = nn.BatchNorm1d(n_enc_3)

        self.z_layer = Linear(n_enc_3, n_z)
        self.BN4 = nn.BatchNorm1d(n_z)

        self.dec_1 = Linear(n_z, n_dec_1)
        self.BN7 = nn.BatchNorm1d(n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.BN8 = nn.BatchNorm1d(n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.BN9 = nn.BatchNorm1d(n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)

    def forward(self, x):
        enc_h1 = F.relu(self.BN1(self.enc_1(x)))
        enc_h2 = F.relu(self.BN2(self.enc_2(enc_h1)))
        enc_h3 = F.relu(self.BN3(self.enc_3(enc_h2)))

        z = self.BN4(self.z_layer(enc_h3))

        dec_h1 = F.relu(self.BN7(self.dec_1(z)))
        dec_h2 = F.relu(self.BN8(self.dec_2(dec_h1)))
        dec_h3 = F.relu(self.BN9(self.dec_3(dec_h2)))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, enc_h1, enc_h2, enc_h3, z, dec_h1, dec_h2, dec_h3


class scASDC(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z, n_clusters, v=1):
        super(scASDC, self).__init__()

        # autoencoder for intra information
        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,

            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,

            n_input=n_input,
            n_z=n_z)

        self.ae.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))

        self.gnn_1 = GNNLayer(n_input, n_enc_1)
        # self.gnn_2 = GNNLayer(n_enc_1, n_enc_2)#
        self.gnn_3 = GNNLayer(n_enc_1, n_z)  #
        # self.gnn_4 = GNNLayer(n_enc_3, n_z)#
        self.gnn_5 = GNNLayer(n_z, n_clusters)
        self.gnn_6 = GNNLayer(n_clusters, n_dec_1)
        self.gnn_7 = GNNLayer(n_dec_1, n_dec_2)
        # self.gnn_8 = GNNLayer(n_dec_2, n_dec_3)
        self.gnn_9 = GNNLayer(n_dec_2, n_input)
        self.attn1 = SelfAttentionWide(n_enc_1)
        self.attn2 = SelfAttentionWide(n_enc_2)
        self.attn3 = SelfAttentionWide(n_enc_1)
        self.attn4 = SelfAttentionWide(n_z)
        self.attn5 = SelfAttentionWide(n_z)

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        self._dec_mean = nn.Sequential(nn.Linear(n_dec_3, n_input), MeanAct())
        self._dec_disp = nn.Sequential(nn.Linear(n_dec_3, n_input), DispAct())
        self._dec_pi = nn.Sequential(nn.Linear(n_dec_3, n_input), nn.Sigmoid())

        # degree
        self.v = v
        self.zinb_loss = ZINBLoss().cuda()

    def forward(self, x, adj):
        x_bar, tra1, tra2, tra3, z, dec_1, dec_2, dec_3 = self.ae(x)

        # GCN Module
        h = self.gnn_1(x, adj)

        h = self.attn3((h + tra2))
        h = h.squeeze(0)
        h = self.gnn_3(h, adj)

        h1 = self.attn5((h + z))
        h1 = h1.squeeze(0)
        h1 = self.gnn_5(h1, adj, active=False)
        predict = F.softmax(h1, dim=1)

        _mean = self._dec_mean(dec_3)
        _disp = self._dec_disp(dec_3)
        _pi = self._dec_pi(dec_3)
        zinb_loss = self.zinb_loss

        h = self.gnn_6(h1, adj)
        h = self.gnn_7(h + dec_1, adj)
        h = self.gnn_9(h + dec_3, adj)
        A_pred = dot_product_decode(h)

        # Dual Self-supervised Module
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x_bar, q, predict, z, h, A_pred, h1, _mean, _disp, _pi, zinb_loss


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def train_scASDC(dataset, X_raw, sf):
    model = scASDC(1000, 1000, 4000, 4000, 1000, 1000,
                 n_input=args.n_input,
                 n_z=args.n_z,
                 n_clusters=args.n_clusters,
                 v=1.0).to(device)
    print(model)
    optimizer = Adam(model.parameters(), lr=args.lr)

    # KNN Graph
    adj = load_graph(args.name, args.k, dataset.x.shape[0])
    adj = adj.to(device)
    data = torch.Tensor(dataset.x).to(device)
    y = dataset.y
    with torch.no_grad():
        _, _, _, _, z, _, _, _ = model.ae(data)

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20, max_iter=1000)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    y_pred_last = y_pred
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    eva(y, y_pred, 'pae')
    for epoch in range(200):
        if epoch % 1 == 0:
            # update_interval
            _, tmp_q, pred, _, h, A_pred, h1, _, _, _, _ = model(data, adj)
            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)

            res1 = tmp_q.cpu().numpy().argmax(1)
            res2 = pred.data.cpu().numpy().argmax(1)
            res3 = p.data.cpu().numpy().argmax(1)
            eva(y, res2, str(epoch) + 'Z')
            if epoch == 199:
                acc, nmi, ari, f1 = eva(y, res2, str(epoch) + 'Z')

        x_bar, q, pred, _, h, A_pred, h1, meanbatch, dispbatch, pibatch, zinb_loss = model(data, adj)

        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        ce_loss = F.kl_div(pred.log(), p, reduction='batchmean')
        re_loss = F.mse_loss(x_bar, data)
        graph_loss = F.mse_loss(h, data)
        re_graphloss = F.kl_div(A_pred.view(-1).log(), adj.to_dense().view(-1), reduction='batchmean')
        X_raw = torch.as_tensor(X_raw).cuda()
        sf = torch.as_tensor(sf).cuda()

        zinb_loss = zinb_loss(X_raw, meanbatch, dispbatch, pibatch, sf)

        loss = 0.1 * kl_loss + 0.01 * ce_loss + re_loss + 0.01 * re_graphloss + 0.01 * graph_loss + 0.5 * zinb_loss
        # print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return acc, nmi, ari, f1


from warnings import simplefilter

if __name__ == "__main__":
    dataname = 'Romanov'
    top_k = 10
    num_clu = 7
    high_genes = 2000
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default=dataname)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n_clusters', default=num_clu, type=int)
    parser.add_argument('--n_z', default=10, type=int)
    parser.add_argument('--pretrain_path', type=str, default='pkl')
    parser.add_argument('--n_input', type=int, default=high_genes)
    args = parser.parse_args()
    #  use cuda
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")
    simplefilter(action='ignore', category=FutureWarning)

    args.pretrain_path = '../datasets/{}.pkl'.format(args.name)

    print(args)
    x, y = prepro('../datasets/{}/data.h5'.format(args.name))
    x = np.ceil(x).astype(int)
    adata = sc.AnnData(x)
    adata.obs['Group'] = y
    adata = normalize_1(adata, copy=True, highly_genes=high_genes, size_factors=True, normalize_input=True,
                        logtrans_input=True)
    count = adata.X
    construct_graph(count, y, 'ncos', name=dataname, topk=top_k)
    dataset = load_data(count, y)
    sf = adata.obs.size_factors
    acc, nmi, ari, f1 = train_scASDC(dataset, count, sf)