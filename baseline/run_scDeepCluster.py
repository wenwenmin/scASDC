from time import time
import math, os

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from scDeepCluster import scDeepCluster
from single_cell_tools import *
import numpy as np
import collections
from sklearn import metrics
import h5py
import scanpy as sc
from preprocess_scdeepclu import read_dataset, normalize
from preprocess import prepro
from evaluation import *
import warnings
import umap
import matplotlib.pyplot as plt
from datetime import datetime

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

if __name__ == "__main__":
    dataname = 'Quake_10x_Limb_Muscle'
    num_clu = 6
    high_genes = 2000

    # for high_genes in high_genes_li:
    # setting the hyper parameters
    import argparse

    parser = argparse.ArgumentParser(
        description='scDeepCluster: model-based deep embedding clustering for single-cell RNA-seq data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # todo记得修改聚类数！
    parser.add_argument('--n_clusters', default=num_clu, type=int,
                        help='number of clusters, 0 means estimating by the Louvain algorithm')
    parser.add_argument('--knn', default=10, type=int,
                        help='number of nearest neighbors, used by the Louvain algorithm')
    parser.add_argument('--resolution', default=.8, type=float,
                        help='resolution parameter, used by the Louvain algorithm, larger value for more number of clusters')
    # todo选择重要基因数量！
    parser.add_argument('--select_genes', default=high_genes, type=int,
                        help='number of selected genes, 0 means using all genes')
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--data_file', default='data.h5')
    # todo最大迭代轮数
    parser.add_argument('--maxiter', default=2000, type=int)
    # todopre_train_epochs
    parser.add_argument('--pretrain_epochs', default=200, type=int)
    parser.add_argument('--gamma', default=1., type=float,
                        help='coefficient of clustering loss')
    parser.add_argument('--sigma', default=2.5, type=float,
                        help='coefficient of random noise')
    parser.add_argument('--update_interval', default=1, type=int)
    parser.add_argument('--tol', default=0.001, type=float,
                        help='tolerance for delta clustering labels to terminate training stage')
    parser.add_argument('--ae_weights', default=None,
                        help='file to pretrained weights, None for a new pretraining')
    parser.add_argument('--save_dir', default='results/scDeepCluster/',
                        help='directory to save model weights during the training stage')
    parser.add_argument('--ae_weight_file', default='AE_weights.pth.tar',
                        help='file name to save model weights after the pretraining stage')
    parser.add_argument('--final_latent_file', default='final_latent_file.txt',
                        help='file name to save final latent representations')
    parser.add_argument('--predict_label_file', default='pred_labels.txt',
                        help='file name to save final clustering labels')
    parser.add_argument('--device', default='cuda')

    args = parser.parse_args()

    x, y = prepro('../datasets/{}/data.h5'.format(dataname))
    x = np.ceil(x).astype(int)
    if args.select_genes > 0:
        importantGenes = geneSelection(x, n=args.select_genes, plot=False)
        x = x[:, importantGenes]

    # preprocessing scRNA-seq read counts matrix  预处理
    adata = sc.AnnData(x, dtype="float64")
    if y is not None:
        adata.obs['Group'] = y

    adata = read_dataset(adata,
                         transpose=False,
                         test_split=False,
                         copy=True)

    adata = normalize(adata,
                      size_factors=True,
                      normalize_input=True,
                      logtrans_input=True,
                      highly_genes=None)

    input_size = adata.n_vars
    print('input_size:', input_size)

    print(args)

    print(adata.X.shape)
    if y is not None:
        print(y.shape)

    # model训练模型
    model = scDeepCluster(input_dim=adata.n_vars, z_dim=32,
                          encodeLayer=[256, 64], decodeLayer=[64, 256], sigma=args.sigma, gamma=args.gamma,
                          device=args.device)

    print(str(model))

    t0 = time()
    if args.ae_weights is None:
        model.pretrain_autoencoder(X=adata.X, X_raw=adata.raw.X, size_factor=adata.obs.size_factors,
                                   batch_size=args.batch_size, epochs=args.pretrain_epochs,
                                   ae_weights=args.ae_weight_file)
    else:
        if os.path.isfile(args.ae_weights):
            print("==> loading checkpoint '{}'".format(args.ae_weights))
            checkpoint = torch.load(args.ae_weights)
            model.load_state_dict(checkpoint['ae_state_dict'])
        else:
            print("==> no checkpoint found at '{}'".format(args.ae_weights))
            raise ValueError

    print('Pretraining time: %d seconds.' % int(time() - t0))

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    acc_li = []
    nmi_li = []
    ari_li = []
    f1_li = []
    for i in range(5):
        if args.n_clusters > 0:
            y_pred, _, _, _, _, latent = model.fit(X=adata.X, X_raw=adata.raw.X, size_factor=adata.obs.size_factors,
                                           n_clusters=args.n_clusters, init_centroid=None,
                                           y_pred_init=None, y=y, batch_size=args.batch_size, num_epochs=args.maxiter,
                                           update_interval=args.update_interval, tol=args.tol, save_dir=args.save_dir)
        print('Total time: %d seconds.' % int(time() - t0))

        if y is not None:
            #    acc = np.round(cluster_acc(y, y_pred), 5)
            nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
            ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
            print('Evaluating cells: NMI= %.4f, ARI= %.4f' % (nmi, ari))
            acc, nmi, ari, f1 = eva(y, y_pred)
            acc_li.append(acc)
            nmi_li.append(nmi)
            ari_li.append(ari)
            f1_li.append(f1)

    np.savez("results/scDeepCluster_{}.npz".format(dataname), ARI=ari_li, NMI=nmi_li, ACC=acc_li, f1=f1_li)
    loaded_data = np.load('results/scDeepCluster_{}.npz'.format(dataname))
    acc1 = loaded_data['ACC']
    nmi1 = loaded_data['NMI']
    ari1 = loaded_data['ARI']
    f11 = loaded_data['f1']

    combined_array = np.column_stack((acc1, nmi1, ari1, f11))

    df = pd.DataFrame(combined_array, columns=['ACC', 'NMI', 'ARI', 'F1'])
    df['Timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    file_path = "results/scDeepCluster_{}_res.csv".format(dataname)