# scASDC
scASDC employs a multi-layer graph convolutional network (GCN) to capture high-order structural relationships between cells, termed as the graph autoencoder module. 
To mitigate the oversmoothing issue in GCNs, we introduce a ZINB-based autoencoder module that extracts content information from the data and learns latent representations of gene expression. These modules are further integrated through an attention fusion mechanism, ensuring effective combination of gene expression and structural information at each layer of the GCN. 
Additionally, a self-supervised learning module is incorporated to enhance the robustness of the learned embeddings. Extensive experiments demonstrate that scASDC outperforms existing state-of-the-art methods, providing a robust and effective solution for single-cell clustering tasks. 
![fig1_1](https://github.com/user-attachments/assets/d05ade1e-8b8b-4371-9682-621dea57d3e0)

# Data Availability
The actual datasets we used can be downloaded from the [data](https://zenodo.org/records/12814320).

