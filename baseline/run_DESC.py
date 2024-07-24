import os

os.environ['PYTHONHASHSEED'] = '0'
import desc
import pandas as pd
import numpy as np
import scanpy as sc
from evaluation_DCA import eva
from time import time
import sys
import matplotlib
# matplotlib.use('TkAgg')  # 或 'Qt5Agg'
import matplotlib.pyplot as plt

from preprocess import prepro, normalize_1
import umap.umap_ as umap
import torch
import random
import tensorflow as tf
from datetime import datetime

sc.settings.set_figure_params(dpi=300)
sc.settings.verbosity = 3
print(sys.version)

graphviz_setup = "D:\graphviz\windows_10_msbuild_Release_graphviz-9.0.0-win32\Graphviz\bin"
os.environ["PATH"] += os.pathsep + graphviz_setup

save_dir = "paul_result_2"

dataname = 'Quake_10x_Limb_Muscle'
num_clu = 6
# high_genes_li = [500, 1000, 1500, 2000]
high_genes = 2000

x, y = prepro('../datasets/{}/data.h5'.format(dataname))
print("Cell number:", x.shape[0])
print("Gene number", x.shape[1])
x = np.ceil(x).astype(int)
cluster_number = len(np.unique(y))
print("Cluster number:", cluster_number)

adata = sc.AnnData(x)
adata.obs['celltype'] = y
adata.obs['celltype2'] = y
adata = normalize_1(adata, copy=True, highly_genes=high_genes, size_factors=True, normalize_input=True,
                    logtrans_input=True)

acc_li = []
nmi_li = []
ari_li = []
f1_li = []
for i in range(5):
    adata, emb_z = desc.train(adata,
                       dims=[adata.shape[1], 64, 32],
                       tol=0.005,
                       n_neighbors=10,
                       batch_size=256,
                       louvain_resolution=[0.8],
                       # not necessarily a list, you can only set one value, like, louvain_resolution=1.0
                       save_dir=str(save_dir),
                       do_tsne=False,
                       learning_rate=200,  # the parameter of tsne
                       use_GPU=True,
                       GPU_id=0,
                       num_Cores=1,  # for reproducible, only use 1 cpu
                       num_Cores_tsne=4,
                       save_encoder_weights=False,
                       save_encoder_step=3,  # save_encoder_weights is False, this parameter is not used
                       use_ae_weights=False,
                       random_seed=(i+1)*2453,
                       do_umap=False)  # if do_uamp is False, it will don't compute umap coordiate

    y_pred = np.asarray(adata.obs['desc_0.8'], dtype=int)
    acc, nmi, ari, f1 = eva(y, y_pred)
    acc_li.append(acc)
    nmi_li.append(nmi)
    ari_li.append(ari)
    f1_li.append(f1)
np.savez("results/DESC_{}.npz".format(dataname), ARI=ari_li, NMI=nmi_li, ACC=acc_li, f1=f1_li)

loaded_data = np.load('results/DESC_{}.npz'.format(dataname))

acc1 = loaded_data['ACC']
nmi1 = loaded_data['NMI']
ari1 = loaded_data['ARI']
f11 = loaded_data['f1']

combined_array = np.column_stack((acc1, nmi1, ari1, f11))
header = 'ACC,NMI,ARI,F1'

df = pd.DataFrame(combined_array, columns=['ACC', 'NMI', 'ARI', 'F1'])
df['Timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# 文件路径
file_path = 'results/DESC_{}_res.csv'.format(dataname)