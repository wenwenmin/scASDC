from __future__ import print_function, division
import argparse
import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear
from utils import load_data, load_graph
from GNN import GNNLayer
from evaluation import eva
from preprocess import normalize_1, prepro
import scanpy as sc
import umap
import matplotlib.pyplot as plt
from collections import Counter


torch.cuda.set_device(0)


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

        return x_bar, enc_h1, enc_h2, enc_h3, z


class SDCN(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z, n_clusters, v=1):
        super(SDCN, self).__init__()

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

        # # GCN for inter information
        # self.gnn_1 = GNNLayer(n_input, n_enc_1)
        # self.gnn_2 = GNNLayer(n_enc_1, n_enc_2)
        # self.gnn_3 = GNNLayer(n_enc_2, n_enc_3)
        # self.gnn_4 = GNNLayer(n_enc_3, n_z)
        # self.gnn_5 = GNNLayer(n_z, n_clusters)

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # degree
        self.v = v

    def forward(self, x):
        # DNN Module
        x_bar, tra1, tra2, tra3, z = self.ae(x)

        # sigma = 0.5
        #
        # # GCN Module
        # h = self.gnn_1(x, adj)
        # h = self.gnn_2((1 - sigma) * h + sigma * tra1, adj)
        # h = self.gnn_3((1 - sigma) * h + sigma * tra2, adj)
        # h = self.gnn_4((1 - sigma) * h + sigma * tra3, adj)
        # h = self.gnn_5((1 - sigma) * h + sigma * z, adj, active=False)
        # predict = F.softmax(h, dim=1)

        # Dual Self-supervised Module
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x_bar, q, z


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def train_sdcn(dataset):
    model = SDCN(1000, 1000, 4000, 4000, 1000, 1000,
                 n_input=args.n_input,
                 n_z=args.n_z,
                 n_clusters=args.n_clusters,
                 v=1.0).to(device)
    print(model)

    optimizer = Adam(model.parameters(), lr=args.lr)

    # cluster parameter initiate
    data = torch.Tensor(dataset.x).to(device)
    y = dataset.y
    with torch.no_grad():
        _, _, _, _, z = model.ae(data)

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    y_pred_last = y_pred
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    eva(y, y_pred, 'pae')

    for epoch in range(200):
        if epoch % 1 == 0:
            # update_interval
            _, tmp_q, pred = model(data)
            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)

            res1 = tmp_q.cpu().numpy().argmax(1)  # Q
            # res2 = pred.data.cpu().numpy().argmax(1)  # Z
            res3 = p.data.cpu().numpy().argmax(1)  # P
            eva(y, res1, str(epoch) + 'Q')
            # eva(y, res2, str(epoch) + 'Z')
            eva(y, res3, str(epoch) + 'P')
            if epoch == 199:
                acc, nmi, ari, f1 = eva(y, res1, str(epoch) + 'Q')

        x_bar, q, _ = model(data)

        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        # ce_loss = F.kl_div(pred.log(), p, reduction='batchmean')
        re_loss = F.mse_loss(x_bar, data)

        loss = 0.1 * kl_loss + re_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return acc, nmi, ari, f1


if __name__ == "__main__":
    dataname = 'Quake_10x_Limb_Muscle'
    num_clu = 6
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
    parser.add_argument('--n_input', type=str, default=high_genes)
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    args.pretrain_path = '../datasets/{}.pkl'.format(args.name)

    print(args)
    x, y = prepro('../datasets/{}/data.h5'.format(dataname))
    x = np.ceil(x).astype(int)
    adata = sc.AnnData(x)
    adata.obs['Group'] = y
    adata = normalize_1(adata, copy=True, highly_genes=high_genes, size_factors=True, normalize_input=True,
                        logtrans_input=True)
    count = adata.X
    dataset = load_data(count, y)
    acc_li = []
    nmi_li = []
    ari_li = []
    f1_li = []
    for i in range(5):
        print('第', i, '次实验')
        print('------------------------------------------------------')
        acc, nmi, ari, f1 = train_sdcn(dataset)
        acc_li.append(acc)
        nmi_li.append(nmi)
        ari_li.append(ari)
        f1_li.append(f1)
    np.savez("results/DEC_{}.npz".format(dataname), ARI=ari_li, NMI=nmi_li, ACC=acc_li, f1=f1_li)
    loaded_data = np.load('results/DEC_{}.npz'.format(dataname))

    acc1 = loaded_data['ACC']
    nmi1 = loaded_data['NMI']
    ari1 = loaded_data['ARI']
    f11 = loaded_data['f1']

    combined_array = np.column_stack((acc1, nmi1, ari1, f11))
    np.savetxt('results/DEC_{}_res.csv'.format(dataname), combined_array, delimiter=',')

