#!/usr/bin/env python3
"""
================================================================================
  🔬 ARISE-Plus v2: Deep Embedding Clustering (DEC) Component Ablation Suite
================================================================================

  Purpose:
  Systematically evaluate which loss terms, spatial consensus priors, and
  latent sub-representations drive the Silhouette score improvement during the
  2-Stage Spatial Potts DEC fine-tuning period.

  Ablation Matrix:
  1. Full_ARISE_v2: Grand Champion baseline (Potts MRF + OT + Dense Gram + Spatial Contrastive + Recon)
  2. wo_Potts_MRF: Standard DEC with lambda_spatial = 0.0 (no neighborhood consensus smoothing)
  3. wo_Sinkhorn_OT: DEC without cross-modal Sinkhorn Optimal Transport loss (L_ot = 0)
  4. wo_Dense_Gram: DEC without relational Gram matrix Frobenius alignment (L_dense = 0)
  5. wo_Spatial_Contrastive: DEC without spatial graph contrastive loss (L_spatial = 0)
  6. wo_Reconstruction: DEC without multi-head autoencoder reconstruction loss (L_recon = 0)
  7. Frozen_Encoder_DEC_Only: Feature encoders frozen during DEC; only cluster centers updated

  Sub-Representation Silhouette Dynamics Tracked:
  - fused_joint (64-dim): Joint multi-modal latent representation
  - fused_rna   (64-dim): Cross-attention refined RNA representation
  - fused_aux   (64-dim): Auxiliary ADT/ATAC representation
  - x_sim       (64-dim): Expression similarity GCN branch
  - x_dist      (64-dim): Spatial Euclidean distance GCN branch

  Authors: FYDP Research Team
  Date: 2026
================================================================================
"""

import os
import sys
import math
import time
import argparse
import random
import warnings
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    homogeneity_score,
    v_measure_score,
    fowlkes_mallows_score,
    silhouette_score
)

warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------
# 1. REPRODUCIBILITY SEEDING
# ----------------------------------------------------------------------

def set_seed(seed: int = 42):
    """Set global random seed for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


# ----------------------------------------------------------------------
# 2. DATA PREPROCESSING & DATASET REGISTRY
# ----------------------------------------------------------------------

BENCHMARK_DATASETS = [
    ("10x_human_lymph_node_A1", "https://drive.google.com/drive/folders/10z1N4MwW8Y49o8GlkYGBKVx1N7fiMuyC"),
    ("10x_human_lymph_node_D1", "https://drive.google.com/drive/folders/1-g_Ca2XMaMXF-MisuVY-wobWDX86O6zz"),
    ("Mouse_Brain_E11_S1", "https://drive.google.com/drive/folders/1zRwDJrYnks0LRzlAVRqPU7jE_OcStgPo"),
    ("Mouse_Brain_E13_S1", "https://drive.google.com/drive/folders/1GOufwIRjjfcd9Bi2GKtebzKoPCg2jVud"),
    ("Mouse_Brain_E15_S1", "https://drive.google.com/drive/folders/1rHkTL5OF5qPsEERypRGMS51SjUQ69tdD"),
    ("Mouse_Brain_E18_S1", "https://drive.google.com/drive/folders/1Xj1LNIAY93biS6JIMKNRODn5GvtCKADB"),
]

def clr_normalize_each_cell(adata, inplace=True):
    def seurat_clr(x):
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)

    if not inplace:
        adata = adata.copy()

    adata.X = np.apply_along_axis(
        seurat_clr, 1, (adata.X.toarray() if sp.issparse(adata.X) else np.array(adata.X))
    )
    return adata

def pca(adata, use_reps=None, n_comps=10):
    from sklearn.decomposition import PCA
    pca_model = PCA(n_components=n_comps)
    if use_reps is not None:
        feat_pca = pca_model.fit_transform(adata.obsm[use_reps])
    else:
        feat_pca = pca_model.fit_transform(adata.X.toarray() if sp.issparse(adata.X) else adata.X)
    return feat_pca

def tfidf(X):
    idf = X.shape[0] / (X.sum(axis=0) + 1e-10)
    if sp.issparse(X):
        tf = X.multiply(1 / (X.sum(axis=1) + 1e-10))
        return sp.csr_matrix(tf.multiply(idf))
    else:
        tf = X / (X.sum(axis=1, keepdims=True) + 1e-10)
        return tf * idf

def preprocess_universal(adata_RNA, adata_omics2, dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
    adata_RNA_copy = adata_RNA.copy()
    sc.pp.filter_genes(adata_RNA_copy, min_cells=10)
    try:
        sc.pp.highly_variable_genes(adata_RNA_copy, flavor="seurat_v3", n_top_genes=3000)
        sc.pp.normalize_total(adata_RNA_copy, target_sum=1e4)
        sc.pp.log1p(adata_RNA_copy)
        sc.pp.scale(adata_RNA_copy)
    except Exception:
        sc.pp.normalize_total(adata_RNA_copy, target_sum=1e4)
        sc.pp.log1p(adata_RNA_copy)
        sc.pp.highly_variable_genes(adata_RNA_copy, flavor="seurat", n_top_genes=3000)
        sc.pp.scale(adata_RNA_copy)

    RNA_expression = adata_RNA_copy[:, adata_RNA_copy.var['highly_variable']].X
    if sp.issparse(RNA_expression):
        RNA_expression = RNA_expression.toarray()

    adata_omics2_copy = adata_omics2[adata_RNA.obs_names].copy()
    if dataset_name.startswith("10x"):
        adata_omics2_copy = clr_normalize_each_cell(adata_omics2_copy)
        sc.pp.scale(adata_omics2_copy)
        omics2_expression = adata_omics2_copy.X
    else:
        adata_omics2_copy.X = tfidf(adata_omics2_copy.X)
        sc.pp.normalize_per_cell(adata_omics2_copy, counts_per_cell_after=1e4)
        sc.pp.log1p(adata_omics2_copy)
        n_comps = min(60, adata_omics2_copy.shape[1])
        adata_omics2_copy.obsm['feat'] = pca(adata_omics2_copy, n_comps=n_comps)
        omics2_expression = adata_omics2_copy.obsm['feat']

    if sp.issparse(omics2_expression):
        omics2_expression = omics2_expression.toarray()

    return RNA_expression, omics2_expression

def load_dataset(dataset_name: str, folder_url: str, base_data_dir: str = "data"):
    base = os.path.join(base_data_dir, dataset_name)
    os.makedirs(base, exist_ok=True)

    rna_path = os.path.join(base, "adata_RNA.h5ad")
    if dataset_name.startswith("10x"):
        other_path = os.path.join(base, "adata_ADT.h5ad")
        annotation_path = os.path.join(base, "annotation.csv")
        gt_col = "manual-anno"
    else:
        other_path = os.path.join(base, "adata_ATAC.h5ad")
        annotation_path = os.path.join(base, "anno.csv")
        gt_col = "cluster"

    if not (os.path.exists(rna_path) and os.path.exists(other_path) and os.path.exists(annotation_path)):
        print(f"Downloading dataset files for {dataset_name}...")
        os.system(f'gdown --folder "{folder_url}" --output "{base}"')

    adata_rna = sc.read_h5ad(rna_path)
    adata_other = sc.read_h5ad(other_path)
    adata_rna.var_names_make_unique()
    adata_other.var_names_make_unique()

    anno_df = pd.read_csv(annotation_path, index_col=0)
    adata_rna.obs['ground_truth'] = anno_df[gt_col]
    adata_other.obs['ground_truth'] = anno_df[gt_col]

    rna_data, aux_data = preprocess_universal(adata_rna, adata_other, dataset_name)
    cell_positions = adata_rna.obsm['spatial']
    num_clusters = adata_rna.obs['ground_truth'].nunique()

    return adata_rna, rna_data, aux_data, cell_positions, num_clusters


# ----------------------------------------------------------------------
# 3. GRAPH TOPOLOGY & BUILDER
# ----------------------------------------------------------------------

def compute_multi_order_motif_matrix(adj_sparse: sp.csr_matrix, k1: float = 1.0, k2: float = 0.5, k3: float = 0.25) -> sp.csr_matrix:
    A_bin = (adj_sparse > 0).astype(np.float32)
    A2 = A_bin.dot(A_bin)
    M3 = A2.multiply(A_bin).tocsr()

    A3 = A2.dot(A_bin)
    M4 = A3.multiply(A_bin).tocsr()

    max_m3 = M3.data.max() if len(M3.data) > 0 else 1.0
    if max_m3 == 0: max_m3 = 1.0
    M3_norm = M3 / max_m3

    max_m4 = M4.data.max() if len(M4.data) > 0 else 1.0
    if max_m4 == 0: max_m4 = 1.0
    M4_norm = M4 / max_m4

    motif_total = k1 * adj_sparse + k2 * M3_norm + k3 * M4_norm
    return motif_total.tocsr()

def build_multimodal_graph_v2(
    x_rna: np.ndarray,
    x_aux: np.ndarray,
    cell_positions: np.ndarray,
    device: str = 'cpu',
    num_neighbors: int = 15
) -> Data:
    num_nodes = x_rna.shape[0]

    # 1. Similarity Graph
    similarity_matrix = cosine_similarity(x_rna)
    nbrs = NearestNeighbors(n_neighbors=num_neighbors + 1, metric='cosine').fit(x_rna)
    _, indices = nbrs.kneighbors(x_rna)

    adj_sim = np.zeros_like(similarity_matrix, dtype=np.float32)
    for i in range(num_nodes):
        for j in indices[i][1:]:
            adj_sim[i, j] = 1.0
            adj_sim[j, i] = 1.0

    sim_sparse = sp.csr_matrix(adj_sim)
    motif_sparse = compute_multi_order_motif_matrix(sim_sparse, k1=1.0, k2=0.5, k3=0.25)

    sim_nonzero = motif_sparse.nonzero()
    sim_edge_index = torch.tensor(np.array(sim_nonzero), dtype=torch.long).to(device)
    sim_edge_weight = torch.tensor(np.array(motif_sparse[sim_nonzero]).flatten(), dtype=torch.float).to(device)

    # 2. Spatial Euclidean Graph
    knn_spatial = kneighbors_graph(cell_positions, n_neighbors=num_neighbors, mode='distance', include_self=False)
    knn_spatial = knn_spatial.maximum(knn_spatial.T)

    dist_edge_index = torch.tensor(knn_spatial.nonzero(), dtype=torch.long).to(device)
    dist_edge_weight = torch.tensor(knn_spatial.data, dtype=torch.float).to(device)

    # 3. Intersection Scaffold
    sim_edges = set(zip(sim_edge_index[0].tolist(), sim_edge_index[1].tolist()))
    dist_edges = set(zip(dist_edge_index[0].tolist(), dist_edge_index[1].tolist()))
    common_edges = sim_edges.intersection(dist_edges)

    if len(common_edges) == 0:
        common_edge_index = dist_edge_index
        common_edge_weight = torch.ones(dist_edge_index.shape[1], dtype=torch.float).to(device)
    else:
        common_edge_index = torch.tensor(list(zip(*common_edges)), dtype=torch.long).to(device)
        common_edge_weight = torch.ones(common_edge_index.shape[1], dtype=torch.float).to(device)

    spatial_adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float, device=device)
    spatial_adj[dist_edge_index[0], dist_edge_index[1]] = 1.0
    spatial_mask = spatial_adj.clone()
    spatial_mask.fill_diagonal_(1.0)

    x_RNA_tensor = torch.tensor(x_rna, dtype=torch.float).to(device)
    x_ADT_tensor = torch.tensor(x_aux, dtype=torch.float).to(device)

    data = Data(
        x_RNA=x_RNA_tensor,
        x_ADT=x_ADT_tensor,
        sim_edge_index=sim_edge_index,
        sim_edge_weight=sim_edge_weight,
        dist_edge_index=dist_edge_index,
        dist_edge_weight=dist_edge_weight,
        common_edge_index=common_edge_index,
        common_edge_weight=common_edge_weight
    )
    data.spatial_mask = spatial_mask
    data.spatial_adj = spatial_adj
    return data


# ----------------------------------------------------------------------
# 4. MODULE DEFINITIONS
# ----------------------------------------------------------------------

class GraphMaskedCrossAttention(nn.Module):
    def __init__(self, d_model: int = 64, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x_query: torch.Tensor, x_key_value: torch.Tensor, spatial_mask: torch.Tensor) -> torch.Tensor:
        N, D = x_query.shape
        residual = x_query

        Q = self.q_proj(x_query).view(N, self.n_heads, self.head_dim).transpose(0, 1)
        K = self.k_proj(x_key_value).view(N, self.n_heads, self.head_dim).transpose(0, 1)
        V = self.v_proj(x_key_value).view(N, self.n_heads, self.head_dim).transpose(0, 1)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask_bool = (spatial_mask == 0).unsqueeze(0)
        scores = scores.masked_fill(mask_bool, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)
        context = context.transpose(0, 1).contiguous().view(N, D)
        out = self.out_proj(context)

        return self.layer_norm(residual + out)


class SinkhornOptimalTransportLoss(nn.Module):
    def __init__(self, eps: float = 0.1, max_iter: int = 30):
        super().__init__()
        self.eps = eps
        self.max_iter = max_iter

    def forward(self, z_rna: torch.Tensor, z_aux: torch.Tensor) -> torch.Tensor:
        N = z_rna.shape[0]
        z_rna_norm = F.normalize(z_rna, p=2, dim=1)
        z_aux_norm = F.normalize(z_aux, p=2, dim=1)

        C = 1.0 - torch.mm(z_rna_norm, z_aux_norm.t())
        mu = torch.full((N,), 1.0 / N, device=z_rna.device, dtype=z_rna.dtype)
        nu = torch.full((N,), 1.0 / N, device=z_aux.device, dtype=z_aux.dtype)

        K = torch.exp(-C / self.eps)
        u = torch.ones(N, device=z_rna.device, dtype=z_rna.dtype)

        for _ in range(self.max_iter):
            v = nu / (torch.matmul(K.t(), u) + 1e-8)
            u = mu / (torch.matmul(K, v) + 1e-8)

        T = u.unsqueeze(1) * K * v.unsqueeze(0)
        return torch.sum(T * C)


class SpatialPottsDEC(nn.Module):
    def __init__(self, num_clusters: int, latent_dim: int, alpha: float = 1.0, lambda_spatial: float = 0.4):
        super().__init__()
        self.num_clusters = num_clusters
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.lambda_spatial = lambda_spatial
        self.cluster_centers = nn.Parameter(torch.Tensor(num_clusters, latent_dim))
        nn.init.xavier_uniform_(self.cluster_centers)

    def compute_q(self, z: torch.Tensor) -> torch.Tensor:
        dist = torch.sum((z.unsqueeze(1) - self.cluster_centers.unsqueeze(0)) ** 2, dim=2)
        q = 1.0 / (1.0 + dist / self.alpha)
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = q / torch.sum(q, dim=1, keepdim=True)
        return q

    def compute_spatial_target_p(self, q: torch.Tensor, spatial_adj: torch.Tensor) -> torch.Tensor:
        weight = q ** 2 / (torch.sum(q, dim=0, keepdim=True) + 1e-8)
        p_dec = weight / (torch.sum(weight, dim=1, keepdim=True) + 1e-8)

        if self.lambda_spatial <= 1e-6:
            return p_dec

        adj_norm = spatial_adj / (torch.sum(spatial_adj, dim=1, keepdim=True) + 1e-8)
        spatial_consensus = torch.mm(adj_norm, q)

        p_spatial = p_dec * torch.exp(self.lambda_spatial * spatial_consensus)
        p_final = p_spatial / (torch.sum(p_spatial, dim=1, keepdim=True) + 1e-8)
        return p_final

    def forward(self, z: torch.Tensor, spatial_adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.compute_q(z)
        p = self.compute_spatial_target_p(q.detach(), spatial_adj)
        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        return q, kl_loss


# ----------------------------------------------------------------------
# 5. ABLATION CONFIGURATION DATACLASS
# ----------------------------------------------------------------------

@dataclass
class AblationConfig:
    name: str
    description: str
    lambda_spatial: float = 0.4
    enable_ot: bool = True
    enable_dense_gram: bool = True
    enable_spatial_contrastive: bool = True
    enable_recon_in_dec: bool = True
    freeze_encoders_in_dec: bool = False

# Standard Ablation Registry
ABLATION_PRESETS: Dict[str, AblationConfig] = {
    'Full_ARISE_v2': AblationConfig(
        name='Full_ARISE_v2',
        description='Full Grand Champion Architecture (Potts MRF + OT + Dense Gram + Spatial + Recon)',
        lambda_spatial=0.4,
        enable_ot=True,
        enable_dense_gram=True,
        enable_spatial_contrastive=True,
        enable_recon_in_dec=True,
        freeze_encoders_in_dec=False
    ),
    'wo_Potts_MRF': AblationConfig(
        name='wo_Potts_MRF',
        description='Standard DEC without Potts MRF Spatial Smoothing (lambda_spatial=0.0)',
        lambda_spatial=0.0,
        enable_ot=True,
        enable_dense_gram=True,
        enable_spatial_contrastive=True,
        enable_recon_in_dec=True,
        freeze_encoders_in_dec=False
    ),
    'wo_Sinkhorn_OT': AblationConfig(
        name='wo_Sinkhorn_OT',
        description='DEC without Entropic Sinkhorn Optimal Transport Loss (L_ot=0 in DEC)',
        lambda_spatial=0.4,
        enable_ot=False,
        enable_dense_gram=True,
        enable_spatial_contrastive=True,
        enable_recon_in_dec=True,
        freeze_encoders_in_dec=False
    ),
    'wo_Dense_Gram': AblationConfig(
        name='wo_Dense_Gram',
        description='DEC without Dense Relational Gram Matrix Frobenius Alignment (L_dense=0 in DEC)',
        lambda_spatial=0.4,
        enable_ot=True,
        enable_dense_gram=False,
        enable_spatial_contrastive=True,
        enable_recon_in_dec=True,
        freeze_encoders_in_dec=False
    ),
    'wo_Spatial_Contrastive': AblationConfig(
        name='wo_Spatial_Contrastive',
        description='DEC without Spatial Graph Contrastive Loss (L_spatial=0 in DEC)',
        lambda_spatial=0.4,
        enable_ot=True,
        enable_dense_gram=True,
        enable_spatial_contrastive=False,
        enable_recon_in_dec=True,
        freeze_encoders_in_dec=False
    ),
    'wo_Reconstruction': AblationConfig(
        name='wo_Reconstruction',
        description='DEC without Multi-Head Reconstruction Loss (L_recon=0 in DEC)',
        lambda_spatial=0.4,
        enable_ot=True,
        enable_dense_gram=True,
        enable_spatial_contrastive=True,
        enable_recon_in_dec=False,
        freeze_encoders_in_dec=False
    ),
    'Frozen_Encoder_DEC_Only': AblationConfig(
        name='Frozen_Encoder_DEC_Only',
        description='Encoders frozen during DEC; only cluster centers updated',
        lambda_spatial=0.4,
        enable_ot=True,
        enable_dense_gram=True,
        enable_spatial_contrastive=True,
        enable_recon_in_dec=True,
        freeze_encoders_in_dec=True
    )
}


# ----------------------------------------------------------------------
# 6. MODEL ARCHITECTURE WITH ABLATION CONTROLS
# ----------------------------------------------------------------------

class ARISEPlusV2AblationModel(nn.Module):
    def __init__(
        self,
        in_rna_dim: int,
        in_aux_dim: int,
        num_clusters: int,
        hidden_dim: int = 512,
        out_dim: int = 64,
        lambda_spatial: float = 0.4
    ):
        super().__init__()

        # Graph Encoders
        self.x_RNA1 = GCNConv(in_rna_dim, hidden_dim)
        self.x_RNA2 = GCNConv(in_rna_dim, hidden_dim)
        self.sim_conv = GCNConv(hidden_dim, out_dim)
        self.dist_conv = GCNConv(hidden_dim, out_dim)
        self.aux_conv = GCNConv(in_aux_dim, out_dim)

        # Intra-Omic Fusion
        self.fusion1 = nn.Sequential(nn.Linear(2 * out_dim, out_dim))

        # Spatial Cross-Attention
        self.cross_attn = GraphMaskedCrossAttention(d_model=out_dim, n_heads=4, dropout=0.1)

        # Joint Inter-Omic Fusion
        self.fusion2 = nn.Sequential(nn.Linear(2 * out_dim, out_dim))

        # Decoders
        self.deconv1 = nn.Linear(out_dim, hidden_dim)
        self.deconv_rna = nn.Linear(hidden_dim, in_rna_dim)
        self.deconv_aux = nn.Linear(hidden_dim, in_aux_dim)
        self.deconv_joint = nn.Linear(hidden_dim, in_rna_dim + in_aux_dim)

        # Objectives
        self.ot_loss = SinkhornOptimalTransportLoss(eps=0.1, max_iter=30)
        self.spatial_potts_dec = SpatialPottsDEC(num_clusters=num_clusters, latent_dim=out_dim, lambda_spatial=lambda_spatial)

        # Homoscedastic Multi-Task Uncertainty Parameters (s_0..s_4)
        self.log_vars = nn.Parameter(torch.zeros(5))

    def set_cluster_centers(self, centers_np: np.ndarray):
        dev = next(self.parameters()).device
        self.spatial_potts_dec.cluster_centers.data = torch.tensor(centers_np, dtype=torch.float, device=dev)

    def freeze_encoders(self):
        for name, param in self.named_parameters():
            if 'spatial_potts_dec' not in name:
                param.requires_grad = False

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, data: Data, compute_q: bool = False) -> Dict[str, torch.Tensor]:
        xs = F.relu(self.x_RNA1(data.x_RNA, data.sim_edge_index, data.sim_edge_weight))
        x_sim = self.sim_conv(xs, data.sim_edge_index, data.sim_edge_weight)

        xd = F.relu(self.x_RNA2(data.x_RNA, data.dist_edge_index, data.dist_edge_weight))
        x_dist = self.dist_conv(xd, data.dist_edge_index, data.dist_edge_weight)

        x_aux = self.aux_conv(data.x_ADT, data.common_edge_index, data.common_edge_weight)

        z_rna = self.fusion1(torch.cat([x_sim, x_dist], dim=1))
        z_rna_refined = self.cross_attn(z_rna, x_aux, data.spatial_mask)
        fused_joint = self.fusion2(torch.cat([z_rna_refined, x_aux], dim=1))

        q, kl_loss = None, None
        if compute_q:
            q, kl_loss = self.spatial_potts_dec(fused_joint, data.spatial_adj)

        return {
            'x_sim': x_sim,
            'x_dist': x_dist,
            'fused_rna': z_rna_refined,
            'fused_aux': x_aux,
            'x_aux': x_aux,
            'fused_joint': fused_joint,
            'embedding': fused_joint,
            'q': q,
            'kl_loss': kl_loss
        }

    def compute_loss(
        self,
        data: Data,
        outputs: Dict[str, torch.Tensor],
        stage: int = 1,
        cfg: Optional[AblationConfig] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        num_nodes = data.x_RNA.shape[0]

        # 1. Multi-Head Reconstruction Loss
        l_rec = F.mse_loss(torch.cat([data.x_RNA, data.x_ADT], dim=1), self.deconv_joint(F.relu(self.deconv1(outputs['fused_joint']))))
        l_sim = F.mse_loss(data.x_RNA, self.deconv_rna(F.relu(self.deconv1(outputs['x_sim']))))
        l_dist = F.mse_loss(data.x_RNA, self.deconv_rna(F.relu(self.deconv1(outputs['x_dist']))))
        l_aux = F.mse_loss(data.x_ADT, self.deconv_aux(F.relu(self.deconv1(outputs['x_aux']))))
        total_recon = l_rec + l_sim + l_dist + l_aux

        # 2. Spatial Regularization Contrastive Loss
        graph_nei = data.spatial_adj
        graph_neg = 1.0 - graph_nei
        emb_norm = F.normalize(outputs['fused_rna'], p=2, dim=1, eps=1e-8)
        sim_mat = torch.sigmoid(torch.matmul(emb_norm, emb_norm.T) - torch.diag_embed(torch.diag(torch.matmul(emb_norm, emb_norm.T))))
        neigh_loss = torch.mul(graph_nei, torch.log(sim_mat + 1e-10)).mean()
        neg_loss = torch.mul(graph_neg, torch.log(1.0 - sim_mat + 1e-10)).mean()
        l_spatial = -(neigh_loss + neg_loss) / 2.0

        # 3. Dense Relational Gram Alignment Loss
        l_emb = torch.mean((outputs['fused_joint'] - outputs['fused_rna']) ** 2) + torch.mean((outputs['fused_joint'] - outputs['fused_aux']) ** 2)
        gram_r = torch.mm(outputs['fused_rna'], outputs['fused_rna'].t())
        gram_a = torch.mm(outputs['fused_aux'], outputs['fused_aux'].t())
        l_rel = torch.norm(gram_r - gram_a, p='fro') / (num_nodes * num_nodes)
        l_dense = l_emb + 0.1 * l_rel

        # 4. Sinkhorn Optimal Transport Loss
        l_ot = self.ot_loss(outputs['fused_rna'], outputs['fused_aux'])

        # Precision weights (exp(-s_m))
        p0 = torch.exp(-self.log_vars[0])
        p1 = torch.exp(-self.log_vars[1])
        p2 = torch.exp(-self.log_vars[2])
        p3 = torch.exp(-self.log_vars[3])
        p4 = torch.exp(-self.log_vars[4])

        if stage == 1:
            total_loss = (0.5 * p0 * total_recon + 0.5 * self.log_vars[0] +
                          0.5 * p1 * l_spatial + 0.5 * self.log_vars[1] +
                          0.5 * p2 * l_dense + 0.5 * self.log_vars[2] +
                          0.5 * p3 * l_ot + 0.5 * self.log_vars[3])
        else:
            # Stage 2: Selective loss inclusion based on AblationConfig
            total_loss = 0.0
            if cfg is None or cfg.enable_recon_in_dec:
                total_loss += 0.5 * p0 * total_recon + 0.5 * self.log_vars[0]
            if cfg is None or cfg.enable_spatial_contrastive:
                total_loss += 0.5 * p1 * l_spatial + 0.5 * self.log_vars[1]
            if cfg is None or cfg.enable_dense_gram:
                total_loss += 0.5 * p2 * l_dense + 0.5 * self.log_vars[2]
            if cfg is None or cfg.enable_ot:
                total_loss += 0.5 * p3 * l_ot + 0.5 * self.log_vars[3]

            if outputs['kl_loss'] is not None:
                l_kl = outputs['kl_loss']
                total_loss += 0.5 * p4 * l_kl + 0.5 * self.log_vars[4]

        loss_dict = {
            'loss_total': float(total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss),
            'loss_recon': float(total_recon.item()),
            'loss_spatial': float(l_spatial.item()),
            'loss_dense': float(l_dense.item()),
            'loss_ot': float(l_ot.item()),
            'loss_kl': float(outputs['kl_loss'].item() if outputs['kl_loss'] is not None else 0.0)
        }
        return total_loss, loss_dict


# ----------------------------------------------------------------------
# 7. EVALUATION & METRICS COMPUTATION
# ----------------------------------------------------------------------

def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, embeddings: Optional[np.ndarray] = None) -> Dict[str, float]:
    y_true_str = np.asarray(y_true).astype(str)
    y_pred_str = np.asarray(y_pred).astype(str)

    ari = float(adjusted_rand_score(y_true_str, y_pred_str))
    nmi = float(normalized_mutual_info_score(y_true_str, y_pred_str))
    ami = float(adjusted_mutual_info_score(y_true_str, y_pred_str))
    homo = float(homogeneity_score(y_true_str, y_pred_str))
    v_meas = float(v_measure_score(y_true_str, y_pred_str))
    fmi = float(fowlkes_mallows_score(y_true_str, y_pred_str))

    sil = 0.0
    if embeddings is not None and len(np.unique(y_pred_str)) > 1:
        try:
            sil = float(silhouette_score(embeddings, y_pred_str))
        except Exception:
            sil = 0.0

    return {
        'ARI': ari,
        'NMI': nmi,
        'AMI': ami,
        'Homogeneity': homo,
        'V-measure': v_meas,
        'FMI': fmi,
        'Silhouette': sil
    }

def run_kmeans_clustering(embeddings: np.ndarray, num_clusters: int, seed: int = 42) -> np.ndarray:
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=seed)
    return kmeans.fit_predict(embeddings)


# ----------------------------------------------------------------------
# 8. TRAINING & ABLATION EXECUTION ENGINE
# ----------------------------------------------------------------------

def run_dec_ablation_experiment(
    cfg: AblationConfig,
    data: Data,
    ground_truth: np.ndarray,
    num_clusters: int,
    epochs: int = 400,
    pretrain_epochs: Optional[int] = None,
    finetune_epochs: Optional[int] = None,
    lr: float = 0.001,
    seed: int = 42,
    device: str = 'cuda',
    verbose: bool = True
) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame]:
    """
    Executes a single ablation experiment, logging epoch-by-epoch Silhouette dynamics
    for both the primary cluster metric and the internal sub-embeddings during DEC.
    """
    start_time = time.time()
    set_seed(seed)
    dev = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    data = data.to(dev)

    in_rna_dim = data.x_RNA.shape[1]
    in_aux_dim = data.x_ADT.shape[1]

    if pretrain_epochs is None and finetune_epochs is None:
        if epochs == 400:
            pretrain_epochs = 250
            finetune_epochs = 150
        else:
            pretrain_epochs = int(epochs * 0.625)
            finetune_epochs = epochs - pretrain_epochs
    elif pretrain_epochs is not None and finetune_epochs is None:
        finetune_epochs = max(epochs - pretrain_epochs, 50)
    elif finetune_epochs is not None and pretrain_epochs is None:
        pretrain_epochs = max(epochs - finetune_epochs, 100)

    model = ARISEPlusV2AblationModel(
        in_rna_dim=in_rna_dim,
        in_aux_dim=in_aux_dim,
        num_clusters=num_clusters,
        hidden_dim=512,
        out_dim=64,
        lambda_spatial=cfg.lambda_spatial
    ).to(dev)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # STAGE 1: Standard Pre-training (identical across ablations for fair DEC evaluation)
    best_pretrain_emb = None
    best_pretrain_sil = -1.0
    best_pretrain_labels = None

    model.train()
    for epoch in range(pretrain_epochs):
        optimizer.zero_grad()
        outputs = model(data, compute_q=False)
        loss, _ = model.compute_loss(data, outputs, stage=1)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            eval_out = model(data, compute_q=False)
            emb = eval_out['embedding'].detach().cpu().numpy()
        model.train()

        pred_labels = run_kmeans_clustering(emb, num_clusters, seed=seed)
        metrics = compute_all_metrics(ground_truth, pred_labels, emb)
        sil = metrics['Silhouette']

        if sil > best_pretrain_sil or best_pretrain_emb is None:
            best_pretrain_sil = sil
            best_pretrain_emb = emb.copy()
            best_pretrain_labels = pred_labels.copy()

    # Initial Cluster Prototype Setup from Pre-trained Space
    kmeans = KMeans(n_clusters=num_clusters, random_state=seed, n_init=20).fit(best_pretrain_emb)
    model.set_cluster_centers(kmeans.cluster_centers_)

    # Measure Initial (Epoch 0 of DEC) State
    initial_dec_sil = best_pretrain_sil
    initial_dec_metrics = compute_all_metrics(ground_truth, best_pretrain_labels, best_pretrain_emb)

    # Freeze encoders if configured
    if cfg.freeze_encoders_in_dec:
        model.freeze_encoders()
        optimizer = torch.optim.Adam([model.spatial_potts_dec.cluster_centers], lr=lr)

    # STAGE 2: DEC Fine-Tuning with Ablation Controls & Fine-Grained Logging
    dec_epoch_records = []
    subcomp_records = []

    best_sil = -1.0
    best_ari_val = -1.0
    best_dec_epoch = 0
    best_embeddings = None
    best_labels = None

    model.train()
    for epoch in range(finetune_epochs):
        optimizer.zero_grad()
        outputs = model(data, compute_q=True)
        loss, loss_dict = model.compute_loss(data, outputs, stage=2, cfg=cfg)
        loss.backward()
        optimizer.step()

        q = outputs['q'].detach()
        pred_labels = torch.argmax(q, dim=1).cpu().numpy()
        emb_joint = outputs['fused_joint'].detach().cpu().numpy()

        metrics = compute_all_metrics(ground_truth, pred_labels, emb_joint)
        sil = metrics['Silhouette']
        ari = metrics['ARI']

        if sil > best_sil:
            best_sil = sil
            best_dec_epoch = epoch + 1
            best_embeddings = emb_joint.copy()
            best_labels = pred_labels.copy()

        if ari > best_ari_val:
            best_ari_val = ari

        # Track DEC Metric Trajectory
        epoch_record = {
            'ablation': cfg.name,
            'seed': seed,
            'dec_epoch': epoch + 1,
            'total_epoch': pretrain_epochs + epoch + 1,
            'loss_total': loss_dict['loss_total'],
            'loss_kl': loss_dict['loss_kl'],
            'loss_recon': loss_dict['loss_recon'],
            'loss_spatial': loss_dict['loss_spatial'],
            'loss_dense': loss_dict['loss_dense'],
            'loss_ot': loss_dict['loss_ot'],
            'ARI': ari,
            'NMI': metrics['NMI'],
            'Silhouette': sil,
            'Homogeneity': metrics['Homogeneity']
        }
        dec_epoch_records.append(epoch_record)

        # Track Silhouette Score across all Sub-Embedding Components
        subcomp_names = ['fused_joint', 'fused_rna', 'fused_aux', 'x_sim', 'x_dist']
        sub_row = {'ablation': cfg.name, 'seed': seed, 'dec_epoch': epoch + 1}
        for cname in subcomp_names:
            c_emb = outputs[cname].detach().cpu().numpy()
            if len(np.unique(pred_labels)) > 1:
                try:
                    c_sil = float(silhouette_score(c_emb, pred_labels))
                except Exception:
                    c_sil = 0.0
            else:
                c_sil = 0.0
            sub_row[cname] = c_sil
        subcomp_records.append(sub_row)

        if (epoch + 1) % 50 == 0 or epoch == 0 or epoch == finetune_epochs - 1:
            if verbose:
                print(f"[{cfg.name} | Seed {seed}] DEC Epoch {epoch+1:3d}/{finetune_epochs} | Loss: {loss_dict['loss_total']:.4f} | KL: {loss_dict['loss_kl']:.4f} | ARI: {ari:.4f} | Sil: {sil:.4f} | ΔSil: {sil - initial_dec_sil:+.4f}")

    elapsed_time = time.time() - start_time
    final_metrics = compute_all_metrics(ground_truth, best_labels, best_embeddings)
    final_metrics['Initial_DEC_Sil'] = float(initial_dec_sil)
    final_metrics['Initial_DEC_ARI'] = float(initial_dec_metrics['ARI'])
    final_metrics['Peak_DEC_Sil'] = float(best_sil)
    final_metrics['Delta_DEC_Sil'] = float(best_sil - initial_dec_sil)
    final_metrics['Peak_DEC_ARI'] = float(best_ari_val)
    final_metrics['Best_DEC_Epoch'] = int(best_dec_epoch)
    final_metrics['Runtime_Sec'] = round(float(elapsed_time), 2)

    df_dec_epochs = pd.DataFrame(dec_epoch_records)
    df_subcomp = pd.DataFrame(subcomp_records)
    return final_metrics, df_dec_epochs, df_subcomp


# ----------------------------------------------------------------------
# 9. VISUALIZATION ENGINE FOR ABLATION ANALYSIS
# ----------------------------------------------------------------------

def generate_ablation_visualizations(
    df_all_epochs: pd.DataFrame,
    df_all_subcomp: pd.DataFrame,
    df_summary: pd.DataFrame,
    out_dir: str,
    dataset_name: str
):
    """
    Produces publication-grade visual comparative trajectory charts:
    1. Silhouette Score Trajectory across DEC Epochs (Comparison across ablations)
    2. Sub-Embedding Component Silhouette Dynamics (Comparison across internal representations)
    3. Net Silhouette Delta Bar Chart (Comparing component contributions)
    """
    os.makedirs(out_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("tab10")

    # 1. Comparative Silhouette Trajectory across Ablations
    plt.figure(figsize=(12, 7), dpi=150)
    sns.lineplot(
        data=df_all_epochs,
        x='dec_epoch',
        y='Silhouette',
        hue='ablation',
        style='ablation',
        linewidth=2.5,
        palette=palette
    )
    plt.title(f"DEC Silhouette Score Trajectory Across Ablations ({dataset_name})", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("DEC Fine-Tuning Epoch", fontsize=12, fontweight='semibold')
    plt.ylabel("Silhouette Score", fontsize=12, fontweight='semibold')
    plt.legend(title="Ablation Setting", bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
    plt.tight_layout()
    sil_path = os.path.join(out_dir, f"{dataset_name}_dec_ablation_silhouette_trajectory.png")
    plt.savefig(sil_path, bbox_inches='tight')
    plt.close()
    print(f"Saved Silhouette trajectory plot to: {sil_path}")

    # 2. Comparative ARI Trajectory across Ablations
    plt.figure(figsize=(12, 7), dpi=150)
    sns.lineplot(
        data=df_all_epochs,
        x='dec_epoch',
        y='ARI',
        hue='ablation',
        style='ablation',
        linewidth=2.5,
        palette=palette
    )
    plt.title(f"DEC ARI Trajectory Across Ablations ({dataset_name})", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("DEC Fine-Tuning Epoch", fontsize=12, fontweight='semibold')
    plt.ylabel("Adjusted Rand Index (ARI)", fontsize=12, fontweight='semibold')
    plt.legend(title="Ablation Setting", bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
    plt.tight_layout()
    ari_path = os.path.join(out_dir, f"{dataset_name}_dec_ablation_ari_trajectory.png")
    plt.savefig(ari_path, bbox_inches='tight')
    plt.close()
    print(f"Saved ARI trajectory plot to: {ari_path}")

    # 3. Sub-Embedding Component Silhouette Dynamics (for Full_ARISE_v2 baseline)
    df_full_sub = df_all_subcomp[df_all_subcomp['ablation'] == 'Full_ARISE_v2']
    if not df_full_sub.empty:
        df_melted = df_full_sub.melt(
            id_vars=['ablation', 'seed', 'dec_epoch'],
            value_vars=['fused_joint', 'fused_rna', 'fused_aux', 'x_sim', 'x_dist'],
            var_name='Component',
            value_name='Silhouette'
        )
        plt.figure(figsize=(11, 6), dpi=150)
        sns.lineplot(
            data=df_melted,
            x='dec_epoch',
            y='Silhouette',
            hue='Component',
            linewidth=2.5,
            palette="Set2"
        )
        plt.title(f"Sub-Embedding Silhouette Dynamics in DEC (Full ARISE-Plus v2 | {dataset_name})", fontsize=14, fontweight='bold', pad=12)
        plt.xlabel("DEC Fine-Tuning Epoch", fontsize=12, fontweight='semibold')
        plt.ylabel("Silhouette Score of Sub-Space", fontsize=12, fontweight='semibold')
        plt.legend(title="Latent Component", frameon=True)
        plt.tight_layout()
        sub_path = os.path.join(out_dir, f"{dataset_name}_dec_subcomponent_silhouette_dynamics.png")
        plt.savefig(sub_path, bbox_inches='tight')
        plt.close()
        print(f"Saved Sub-Component Silhouette dynamics plot to: {sub_path}")

    # 4. Net Silhouette Gain (Delta Silhouette = Peak_DEC_Sil - Initial_DEC_Sil)
    if 'Delta_DEC_Sil' in df_summary.columns:
        plt.figure(figsize=(10, 5), dpi=150)
        agg_delta = df_summary.groupby('ablation')['Delta_DEC_Sil'].mean().reset_index()
        agg_delta = agg_delta.sort_values(by='Delta_DEC_Sil', ascending=False)
        colors = ['#10b981' if v >= 0 else '#ef4444' for v in agg_delta['Delta_DEC_Sil']]
        sns.barplot(data=agg_delta, x='ablation', y='Delta_DEC_Sil', palette=colors)
        plt.axhline(0, color='gray', linestyle='--', alpha=0.7)
        plt.title(f"Net DEC Silhouette Gain (Δ Silhouette) by Component ({dataset_name})", fontsize=13, fontweight='bold', pad=12)
        plt.xlabel("Ablation Setting", fontsize=11, fontweight='semibold')
        plt.ylabel("Δ Silhouette (Peak - Initial)", fontsize=11, fontweight='semibold')
        plt.xticks(rotation=25, ha='right')
        plt.tight_layout()
        bar_path = os.path.join(out_dir, f"{dataset_name}_dec_delta_silhouette_comparison.png")
        plt.savefig(bar_path, bbox_inches='tight')
        plt.close()
        print(f"Saved Delta Silhouette bar plot to: {bar_path}")


# ----------------------------------------------------------------------
# 10. CLI ENTRYPOINT & ORCHESTRATION
# ----------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="ARISE-Plus v2: DEC Component Ablation Suite")
    parser.add_argument('--dataset', type=str, default='0', help="Dataset index (0-5), comma-separated list, or 'all'")
    parser.add_argument('--data_dir', type=str, default='data', help="Local data directory")
    parser.add_argument('--seeds', nargs='+', type=int, default=[42], help="Random seeds (default: [42])")
    parser.add_argument('--epochs', type=int, default=400, help="Total epochs (default: 400)")
    parser.add_argument('--pretrain_epochs', type=int, default=None, help="Explicit pretrain epochs (default: 250)")
    parser.add_argument('--finetune_epochs', type=int, default=None, help="Explicit DEC finetune epochs (default: 150)")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--ablations', type=str, default='all', help="Comma-separated ablation names or 'all'")
    parser.add_argument('--out_dir', type=str, default='results_dec_ablation', help="Output directory for ablation logs and plots")
    parser.add_argument('--plot_only', action='store_true', help="Regenerate all ablation plots from saved CSVs without training")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    summary_csv_path = os.path.join(args.out_dir, "dec_ablation_summary.csv")
    epochs_csv_path = os.path.join(args.out_dir, "dec_ablation_epoch_dynamics.csv")
    subcomp_csv_path = os.path.join(args.out_dir, "dec_subcomponent_silhouette_dynamics.csv")

    # If --plot_only is requested, load existing CSV files and regenerate plots
    if args.plot_only:
        print("=" * 80)
        print("  📊 ARISE-Plus v2: Regenerating Plots from Saved Epoch Data")
        print(f"  Source Directory: {args.out_dir}")
        print("=" * 80)

        if not (os.path.exists(summary_csv_path) and os.path.exists(epochs_csv_path) and os.path.exists(subcomp_csv_path)):
            print(f"❌ Error: Required CSV logs not found in {args.out_dir}!")
            print(f"Expected files: \n - {summary_csv_path}\n - {epochs_csv_path}\n - {subcomp_csv_path}")
            return

        df_summary = pd.read_csv(summary_csv_path)
        df_epochs = pd.read_csv(epochs_csv_path)
        df_subcomp = pd.read_csv(subcomp_csv_path)

        datasets = df_epochs['dataset'].unique()
        for ds_name in datasets:
            print(f"\n---> Generating plots for dataset: {ds_name}")
            generate_ablation_visualizations(
                df_all_epochs=df_epochs[df_epochs['dataset'] == ds_name],
                df_all_subcomp=df_subcomp[df_subcomp['dataset'] == ds_name],
                df_summary=df_summary[df_summary['dataset'] == ds_name],
                out_dir=args.out_dir,
                dataset_name=ds_name
            )
        print("\n[Done] All plots regenerated successfully from per-epoch CSV data.")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Filter target datasets
    if args.dataset == 'all':
        target_datasets = BENCHMARK_DATASETS
    else:
        indices = [int(i.strip()) for i in args.dataset.split(',') if i.strip().isdigit()]
        target_datasets = [BENCHMARK_DATASETS[i] for i in indices if i < len(BENCHMARK_DATASETS)]

    # Filter target ablations
    if args.ablations == 'all':
        target_ablations = list(ABLATION_PRESETS.values())
    else:
        ablation_names = [a.strip() for a in args.ablations.split(',')]
        target_ablations = [ABLATION_PRESETS[name] for name in ablation_names if name in ABLATION_PRESETS]

    print("=" * 80)
    print("  🔬 ARISE-Plus v2: DEC Component Ablation Suite")
    print(f"  Device: {device.upper()} | Seeds: {args.seeds} | Pretrain: {args.pretrain_epochs or 250} | DEC: {args.finetune_epochs or 150}")
    print(f"  Target Ablations ({len(target_ablations)}): {[a.name for a in target_ablations]}")
    print(f"  Output Directory: {args.out_dir}")
    print("=" * 80)

    summary_rows = []
    all_dec_epochs_list = []
    all_subcomp_list = []
    total_start = time.time()

    for ds_idx, (ds_name, ds_url) in enumerate(target_datasets):
        print(f"\n{'='*80}\n========== DATASET: {ds_name} ==========\n{'='*80}")
        adata_rna, rna_data, aux_data, cell_positions, num_clusters = load_dataset(ds_name, ds_url, args.data_dir)
        ground_truth = np.array(adata_rna.obs['ground_truth'].astype('category').cat.codes)

        # Build Graph Data once per dataset
        graph_data = build_multimodal_graph_v2(rna_data, aux_data, cell_positions, device=device, num_neighbors=15)

        for ablation_cfg in target_ablations:
            print(f"\n---> Running Ablation: [{ablation_cfg.name}] : {ablation_cfg.description}")
            for seed in args.seeds:
                metrics, df_dec_ep, df_sub = run_dec_ablation_experiment(
                    cfg=ablation_cfg,
                    data=graph_data,
                    ground_truth=ground_truth,
                    num_clusters=num_clusters,
                    epochs=args.epochs,
                    pretrain_epochs=args.pretrain_epochs,
                    finetune_epochs=args.finetune_epochs,
                    lr=args.lr,
                    seed=seed,
                    device=device,
                    verbose=True
                )

                df_dec_ep['dataset'] = ds_name
                df_sub['dataset'] = ds_name
                all_dec_epochs_list.append(df_dec_ep)
                all_subcomp_list.append(df_sub)

                res_row = {
                    'dataset': ds_name,
                    'ablation': ablation_cfg.name,
                    'seed': seed,
                    **metrics
                }
                summary_rows.append(res_row)

        # Incrementally save and Plot per Dataset
        df_ds_summary = pd.DataFrame(summary_rows)
        df_ds_epochs = pd.concat(all_dec_epochs_list, ignore_index=True)
        df_ds_subcomp = pd.concat(all_subcomp_list, ignore_index=True)

        # Save per-dataset CSV files
        ds_prefix = os.path.join(args.out_dir, ds_name)
        df_ds_summary[df_ds_summary['dataset'] == ds_name].to_csv(f"{ds_prefix}_dec_summary.csv", index=False)
        df_ds_epochs[df_ds_epochs['dataset'] == ds_name].to_csv(f"{ds_prefix}_dec_ablation_epoch_dynamics.csv", index=False)
        df_ds_subcomp[df_ds_subcomp['dataset'] == ds_name].to_csv(f"{ds_prefix}_dec_subcomponent_silhouette_dynamics.csv", index=False)

        # Also write the master consolidated CSVs incrementally
        df_ds_summary.to_csv(summary_csv_path, index=False)
        df_ds_epochs.to_csv(epochs_csv_path, index=False)
        df_ds_subcomp.to_csv(subcomp_csv_path, index=False)

        generate_ablation_visualizations(
            df_all_epochs=df_ds_epochs[df_ds_epochs['dataset'] == ds_name],
            df_all_subcomp=df_ds_subcomp[df_ds_subcomp['dataset'] == ds_name],
            df_summary=df_ds_summary[df_ds_summary['dataset'] == ds_name],
            out_dir=args.out_dir,
            dataset_name=ds_name
        )

    total_time = time.time() - total_start
    print("\n" + "#" * 80)
    print("################### DEC COMPONENT ABLATION SUMMARY ###################")
    print("#" * 80)
    df_final_summary = pd.DataFrame(summary_rows)
    print(df_final_summary[['dataset', 'ablation', 'seed', 'Initial_DEC_Sil', 'Peak_DEC_Sil', 'Delta_DEC_Sil', 'Peak_DEC_ARI', 'Best_DEC_Epoch']])
    print(f"\n[Done] All logs saved incrementally to:\n - {summary_csv_path}\n - {epochs_csv_path}\n - {subcomp_csv_path}")
    print(f"Per-dataset epoch logs also saved to: {args.out_dir}/<dataset_name>_dec_ablation_epoch_dynamics.csv")
    print(f"Total Execution Time: {total_time/60:.2f} min ({total_time:.1f}s)")

if __name__ == '__main__':
    main()
