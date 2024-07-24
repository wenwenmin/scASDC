from preprocess import prepro, normalize_1
import scanpy as sc
import warnings
import numpy as np
from dca.api import dca
from evaluation_DCA import eva
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import os

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
dataname = 'Quake_10x_Limb_Muscle'
num_clu = 6
high_genes = 2000
high_genes_li = [500, 1000, 1500, 2000]

x, y = prepro('../datasets/{}/data.h5'.format(dataname))
x = np.ceil(x).astype(int)
adata = sc.AnnData(x)
print(adata)
adata.obs['Group'] = y
sc.pp.filter_genes(adata, min_counts=1)
sc.pp.filter_cells(adata, min_counts=1)
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
# sc.pp.scale(adata)
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=high_genes,
                                    subset=True)
sc.pp.filter_genes(adata, min_counts=1)
acc_li = []
nmi_li = []
ari_li = []
f1_li = []
for i in range(5):
    adata1 = dca(adata, mode='denoise', ae_type='nb-conddisp', normalize_per_cell=False, scale=False, return_model=False,
                log1p=False, hidden_size=(64, 32, 64), batch_size=32, return_info=True, check_counts=False, copy=True)
    print(adata1)

    # pca降维
    X_new = StandardScaler().fit_transform(adata1.X)
    print(X_new)
    pca = PCA(n_components=2)
    pca.fit(X_new)
    X = pca.transform(X_new)
    print(X)

    y_predict = KMeans(n_clusters=num_clu, n_init=20, max_iter=1000).fit_predict(X)

    acc, nmi, ari, f1 = eva(y, y_predict)
    acc_li.append(acc)
    nmi_li.append(nmi)
    ari_li.append(ari)
    f1_li.append(f1)
np.savez("results/DCA_{}.npz".format(dataname), ARI=ari_li, NMI=nmi_li, ACC=acc_li, f1=f1_li)

loaded_data = np.load('results/DCA_{}.npz'.format(dataname))

acc1 = loaded_data['ACC']
nmi1 = loaded_data['NMI']
ari1 = loaded_data['ARI']
f11 = loaded_data['f1']

combined_array = np.column_stack((acc1, nmi1, ari1, f11))
df = pd.DataFrame(combined_array, columns=['ACC', 'NMI', 'ARI', 'F1'])
df['Timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

file_path = 'results/DCA_{}_res.csv'.format(dataname)