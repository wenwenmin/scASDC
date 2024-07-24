import torch
import torch.optim.lr_scheduler as lr_scheduler
import utils_AttentionAE as utils
import numpy as np
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score, davies_bouldin_score
import time
import scanpy as sc
import random
from preprocess import prepro, normalize_1
from loss_AttentionAE import ZINBLoss
import umap
import matplotlib.pyplot as plt

from scipy.special import digamma
from tqdm import trange
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Function, Variable

_epsilon = 1e-6
random.seed(1)


def train(init_model, Zscore_data, rawData, adj, r_adj, size_factor, device, args):
    start_time = time.time()

    start_mem = torch.cuda.max_memory_allocated(device=device)

    init_model.to(device)
    data = torch.Tensor(Zscore_data).to(device)
    sf = torch.autograd.Variable((torch.from_numpy(size_factor[:, None]).type(torch.FloatTensor)).to(device),
                                 requires_grad=True)
    optimizer = torch.optim.Adam(init_model.parameters(), lr=args.lr)
    adj = torch.Tensor(adj).to(device)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5, last_epoch=-1)
    best_model = init_model
    loss_update = 100000
    for epoch in range(args.training_epoch):
        z, A_pred, pi, mean, disp = init_model(data, adj)
        l = ZINBLoss(theta_shape=(args.n_input,))
        zinb_loss = l(mean * sf, pi, target=torch.tensor(rawData).to(device), theta=disp)
        re_graphloss = torch.nn.functional.mse_loss(A_pred.view(-1), torch.Tensor(r_adj).to(device).view(-1))
        loss = zinb_loss + 0.1 * re_graphloss

        if (epoch + 1) % 10 == 0:
            print("epoch %d, loss %.4f, zinb_loss %.4f, re_graphloss %.4f"
                  % (epoch + 1, loss, zinb_loss, re_graphloss))

        if loss_update > loss:
            loss_update = loss
            best_model = init_model
            epoch_update = epoch

        if ((epoch - epoch_update) > 50):
            print("Early stopping at epoch {}".format(epoch_update))
            elapsed_time = time.time() - start_time
            max_mem = torch.cuda.max_memory_allocated(device=device) - start_mem
            print("Finish Training! Elapsed time: {:.4f} seconds, Max memory usage: {:.4f} MB".format(elapsed_time,
                                                                                                      max_mem / 1024 / 1024))
            return best_model
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(init_model.parameters(), max_norm=3, norm_type=2)
        optimizer.step()
        scheduler.step()
    elapsed_time = time.time() - start_time
    max_mem = torch.cuda.max_memory_allocated(device=device) - start_mem
    print("Finish Training! Elapsed time: {:.4f} seconds, Max memory usage: {:.4f} MB".format(elapsed_time,
                                                                                              max_mem / 1024 / 1024))
    return best_model, elapsed_time


alpha = 1


def loss_func(z, cluster_layer):
    q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - cluster_layer) ** 2, dim=2) / alpha)
    q = q ** (alpha + 1.0) / 2.0
    q = (q.t() / torch.sum(q, dim=1)).t()

    weight = q ** 2 / torch.sum(q, dim=0)
    p = (weight.t() / torch.sum(weight, dim=1)).t()

    log_q = torch.log(q)
    loss = torch.nn.functional.kl_div(log_q, p, reduction='batchmean')
    return loss, p


def clustering(pretrain_model, Zscore_data, rawData, celltype, adj, r_adj, size_factor, device, args):
    start_time = time.time()

    start_mem = torch.cuda.max_memory_allocated(device=device)

    data = torch.Tensor(Zscore_data).to(device)
    adj = torch.Tensor(adj).to(device)
    model = pretrain_model.to(device)
    sf = torch.autograd.Variable((torch.from_numpy(size_factor[:, None]).type(torch.FloatTensor)).to(device),
                                 requires_grad=True)
    # cluster center
    with torch.no_grad():
        z, _, _, _, _ = model(data, adj)

    cluster_centers, init_label = utils.use_Leiden(z.detach().cpu().numpy(), resolution=args.resolution)
    cluster_layer = torch.autograd.Variable((torch.from_numpy(cluster_centers).type(torch.FloatTensor)).to(device),
                                            requires_grad=True)
    asw = np.round(silhouette_score(z.detach().cpu().numpy(), init_label), 3)
    if celltype is not None:
        nmi = np.round(normalized_mutual_info_score(celltype, init_label), 3)
        ari = np.round(adjusted_rand_score(celltype, init_label), 3)
        print('init: ASW= %.3f, ARI= %.3f, NMI= %.3f' % (asw, ari, nmi))
    else:
        print('init: ASW= %.3f' % (asw))

    optimizer = torch.optim.Adam(list(model.enc_1.parameters()) + list(model.enc_2.parameters()) +
                                 list(model.attn1.parameters()) + list(model.attn2.parameters()) +
                                 list(model.gnn_1.parameters()) + list(model.gnn_2.parameters()) +
                                 list(model.z_layer.parameters()) + [cluster_layer], lr=0.001)

    for epoch in range(args.clustering_epoch):
        z, A_pred, pi, mean, disp = model(data, adj)
        kl_loss, ae_p = loss_func(z, cluster_layer)
        l = ZINBLoss(theta_shape=(args.n_input,))
        zinb_loss = l(mean * sf, pi, target=torch.tensor(rawData).to(device), theta=disp)
        re_graphloss = torch.nn.functional.mse_loss(A_pred.view(-1), torch.Tensor(r_adj).to(device).view(-1))
        loss = kl_loss + 0.1 * zinb_loss + 0.01 * re_graphloss
        loss.requires_grad_(True)
        label = utils.dist_2_label(ae_p)

        asw = silhouette_score(z.detach().cpu().numpy(), label)
        db = davies_bouldin_score(z.detach().cpu().numpy(), label)
        # ari = adjusted_rand_score(celltype, label)
        # nmi = normalized_mutual_info_score(celltype, label)

        if (epoch + 1) % 10 == 0:
            print("epoch %d, loss %.4f, kl_loss %.4f, ASW %.3f" % (epoch + 1, loss, kl_loss, asw))
        num = data.shape[0]
        tol = 1e-3
        if epoch == 0:
            last_label = label
        else:
            delta_label = np.sum(label != last_label).astype(np.float32) / num
            last_label = label
            if epoch > 20 and delta_label < tol:
                print('delta_label ', delta_label, '< tol ', tol)
                print("Reach tolerance threshold. Stopping training.")
                elapsed_time = time.time() - start_time
                max_mem = torch.cuda.max_memory_allocated(device=device) - start_mem
                print("Elapsed time: {:.4f} seconds, Max memory usage: {:.4f} MB".format(elapsed_time,
                                                                                         max_mem / 1024 / 1024))
                break
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3, norm_type=2)
        optimizer.step()
    elapsed_time = time.time() - start_time
    max_mem = torch.cuda.max_memory_allocated(device=device) - start_mem
    print("Finish Clustering! Elapsed time: {:.4f} seconds, Max memory usage: {:.4f} MB".format(elapsed_time,
                                                                                                max_mem / 1024 / 1024))

    return [asw, db], last_label, cluster_layer, model, elapsed_time, z


if __name__ == "__main__":
    from warnings import simplefilter
    import random
    from sklearn import preprocessing
    from run_AttentionAE_sc import AttentionAE
    import pandas as pd
    import argparse

    high_genes = 2000

    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default:1e-3')
    parser.add_argument('--n_z', type=int, default=16,
                        help='the number of dimension of latent vectors for each cell, default:16')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='the number of dattention heads, default:8')
    parser.add_argument('--n_hvg', type=int, default=high_genes,
                        help='the number of the highly variable genes, default:2500')
    parser.add_argument('--training_epoch', type=int, default=200,
                        help='epoch of train stage, default:200')
    parser.add_argument('--clustering_epoch', type=int, default=100,
                        help='epoch of clustering stage, default:100')
    parser.add_argument('--resolution', type=float, default=1.0,
                        help='''the resolution of Leiden. The smaller the settings to get the more clusters
                        , advised to 0.1-1.0, default:1.0''')
    # todo什么叫构建细胞连接的方法呀？
    parser.add_argument('--connectivity_methods', type=str, default='gauss',
                        help='method for constructing the cell connectivity ("gauss" or "umap"), default:gauss')
    parser.add_argument('--n_neighbors', type=int, default=10,
                        help='''If True, use a hard threshold to restrict the number of neighbors to n_neighbors, 
                        that is, consider a knn graph. Otherwise, use a Gaussian Kernel to assign low weights to 
                        neighbors more distant than the n_neighbors nearest neighbor. default:100''')
    # todo这里false表示给比n_neighbors更远的邻居分配低权重！True则表示只给n_neighbors分配权重！
    parser.add_argument('--knn', type=int, default=False,
                        help='''If True, use a hard threshold to restrict the number of neighbors to n_neighbors, 
                        that is, consider a knn graph. Otherwise, use a Gaussian Kernel to assign low weights to
                        neighbors more distant than the n_neighbors nearest neighbor. default:False''')
    parser.add_argument('--name', type=str, default=data_name,
                        help='name of input file(a h5ad file: Contains the raw count matrix "X",)')
    # todo这里要检查一下自己的数据里有没有adata.obs["celltype"]！ 没有问题
    parser.add_argument('--celltype', type=str, default='known',
                        help='the true labels of datasets are placed in adata.obs["celltype"]')
    parser.add_argument('--save_pred_label', type=str, default=True,
                        help='To choose whether saves the pred_label to the dict "./pred_label"')
    parser.add_argument('--save_model_para', type=str, default=True,
                        help='To choose whether saves the model parameters to the dict "./model_save"')
    parser.add_argument('--save_embedding', type=str, default=True,
                        help='To choose whether saves the cell embedding to the dict "./embedding"')
    parser.add_argument('--save_umap', type=str, default=True,
                        help='To choose whether saves the visualization to the dict "./umap_figure"')
    # 最大细胞数，4000
    parser.add_argument('--max_num_cell', type=int, default=4000,
                        help='''a maximum threshold about the number of cells use in the model building, 
                        4,000 is the maximum cells that a GPU owning 8 GB memory can handle. 
                        More cells will bemploy the down-sampling straegy, 
                        which has been shown to be equally effective,
                        but it's recommended to process data with less than 24,000 cells at a time''')
    parser.add_argument('--cuda', type=bool, default=True,
                        help='use GPU, or else use cpu (setting as "False")')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    simplefilter(action='ignore', category=FutureWarning)
    for i in ['Quake_10x_Limb_Muscle',
              'Quake_Smart-seq2_Diaphragm', 'Quake_Smart-seq2_Limb_Muscle',
              'Quake_Smart-seq2_Trachea', 'Romanov','Adam']:
        args.name = i
        ASW = []
        ARI = []
        NMI = []
        DB = []
        x, y = prepro('../datasets/{}/data.h5'.format(args.name))
        x = np.ceil(x).astype(int)
        adata0 = sc.AnnData(x)
        adata0.obs['Group'] = y

        adata, rawData, dataset, adj, r_adj = utils.load_data(adata0, args=args, high_g=high_genes)
        celltype = adata.obs['Group']
        for j in range(5):
            print('{}数据集的第{}次实验！-------------------------------------------------'.format(args.name, j))
            if adata.shape[0] < args.max_num_cell:
                random.seed(1000 * j)
                size_factor = adata.obs['size_factors'].values
                Zscore_data = preprocessing.scale(dataset)

                args.n_input = dataset.shape[1]
                print(args)
                init_model = AttentionAE(256, 64, 64, 256, n_input=args.n_input, n_z=args.n_z, heads=args.n_heads,
                                         device=device)
                pretrain_model, _ = train(init_model, Zscore_data, rawData, adj, r_adj, size_factor, device, args)

                metirc, pred_label, _, _, _, emb = clustering(pretrain_model, Zscore_data, rawData, celltype, adj, r_adj,
                                                         size_factor, device, args)
                asw = metirc[0]
                db = metirc[1]

                ari = adjusted_rand_score(celltype, pred_label)
                nmi = normalized_mutual_info_score(celltype, pred_label)
                DB.append(db)
                ASW.append(asw)
                ARI.append(ari)
                NMI.append(nmi)
                print("Final ASW %.3f, ARI %.3f, NMI %.3f" % (asw, ari, nmi))

            # down-sampling input
            else:
                new_adata = utils.random_downsimpling(adata, args.max_num_cell)
                new_adj, new_r_adj = utils.adata_knn(new_adata, method=args.connectivity_methods,
                                                     knn=args.knn, n_neighbors=args.n_neighbors)
                try:
                    new_Zscore_data = preprocessing.scale(new_adata.X.toarray())
                    new_rawData = new_adata.raw[:, adata.raw.var['highly_variable']].X.toarray()
                except:
                    new_Zscore_data = preprocessing.scale(new_adata.X)
                    new_rawData = new_adata.raw[:, adata.raw.var['highly_variable']].X

                size_factor = new_adata.obs['size_factors'].values

                try:
                    Zscore_data = preprocessing.scale(dataset.toarray())

                except:
                    Zscore_data = preprocessing.scale(dataset)

                new_celltype = new_adata.obs['celltype']
                args.n_input = dataset.shape[1]
                print(args)
                init_model = AttentionAE(256, 64, 64, 256, n_input=args.n_input, n_z=args.n_z, heads=args.n_heads,
                                         device=device)
                pretrain_model = train(init_model, new_Zscore_data, new_rawData,
                                       new_adj, new_r_adj, size_factor, device, args)
                _, _, cluster_layer, model, _, emb = clustering(pretrain_model, new_Zscore_data, new_rawData,
                                                           new_celltype, new_adj, new_r_adj, size_factor, device, args)

                data = torch.Tensor(Zscore_data).to(device)
                adj = torch.Tensor(adj).to(device)
                with torch.no_grad():
                    z, _, _, _, _ = model(data, adj)
                    _, p = loss_func(z, cluster_layer)
                    pred_label = utils.dist_2_label(p)
                    asw = np.round(silhouette_score(z.detach().cpu().numpy(), pred_label), 3)
                    nmi = np.round(normalized_mutual_info_score(celltype, pred_label), 3)
                    ari = np.round(adjusted_rand_score(celltype, pred_label), 3)
                    ASW.append(asw)
                    ARI.append(ari)
                    NMI.append(nmi)
                    print("Final ASW %.3f, ARI %.3f, NMI %.3f" % (asw, ari, nmi))
        if i == 'Quake_10x_Limb_Muscle':
            df_ari = pd.DataFrame(ARI, index=range(5), columns=[args.name, ])
            df_asw = pd.DataFrame(ASW, index=range(5), columns=[args.name, ])
            df_nmi = pd.DataFrame(NMI, index=range(5), columns=[args.name, ])
            df_db = pd.DataFrame(DB, index=range(5), columns=[args.name, ])
        else:
            df_ari['%s' % (args.name)] = ARI
            df_asw['%s' % (args.name)] = ASW
            df_nmi['%s' % (args.name)] = NMI
            df_db['%s' % (args.name)] = DB
        # df_ari.to_csv('./results/{}_ARI.csv'.format(args.name))
        # df_asw.to_csv('./results/{}_ASW.csv'.format(args.name))
        # df_nmi.to_csv('./results/{}_NMI.csv'.format(args.name))
        # df_db.to_csv('./results/{}_DB.csv'.format(args.name))
        df_ari.to_csv('./results_AttentionAE/Leiden_ARI_fig.csv')
        # df_asw.to_csv('./results_AttentionAE/Leiden_ASW_fig.csv')
        df_nmi.to_csv('./results_AttentionAE/Leiden_NMI_fig.csv')
        # df_db.to_csv('./results_AttentionAE/Leiden_DB_fig.csv')
        # np.savetxt('./results/%s_predicted_label.csv'%(args.name),pred_label)
