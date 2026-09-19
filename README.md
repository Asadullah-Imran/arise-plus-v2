# 🧬 ARISE-Plus v2: Grand Champion Spatial Multi-Omics Architecture

This folder contains the complete, self-contained standalone implementation of **ARISE-Plus v2**, the state-of-the-art synergistic spatial multi-omics framework.

---

## 🧬 Which Ablation Variants are Combined to Create ARISE-Plus v2?

**ARISE-Plus v2** is a champion architecture designed by distilling and synthesizing the highest-performing empirical mechanisms across the entire ablation study suite (**V0 through V14**):

| Source Variant | Component Contributed to ARISE-Plus v2 | Purpose & Mathematical Formulation |
|:---|:---|:---|
| **`V0` (ARISE Baseline)** | **RNA-Anchored Intersection Scaffold Graph** ($G_{comm}' = G_{sim}' \cap G_{dist}$) | Transfers high-confidence spatial proximity from transcriptomics to noisy auxiliary modalities (ATAC / ADT), eliminating spatial noise and boosting baseline ARI by **+77.4%**. |
| **`V9` (Multi-Order Motifs)** | **Multi-Order 3-Node + 4-Node Clique Topology** ($M_3 + M_4$) | Captures higher-order microenvironmental cohesion via triangular cliques and 4-node cycles: <br> $\mathbf{M}_3 = (\mathbf{A}\cdot\mathbf{A})\odot\mathbf{A}, \quad \mathbf{M}_4 = (\mathbf{A}\cdot\mathbf{A}\cdot\mathbf{A})\odot\mathbf{A}$ <br> $\mathbf{G}_{sim}' = k_1 \mathbf{A} + k_2 \frac{\mathbf{M}_3}{\max \mathbf{M}_3} + k_3 \frac{\mathbf{M}_4}{\max \mathbf{M}_4}$ |
| **`V11` (Spatial Cross-Attention)** | **Graph-Masked Localized Cross-Attention** ($Q=\mathbf{z}_{RNA}, K=\mathbf{x}_{aux}, V=\mathbf{x}_{aux}$) | Spatially constrains cross-modal query-key-value attention to 1-hop physical neighborhood graph edges ($\mathbf{M}_{spatial}$), extracting cross-modal dependencies while avoiding global over-smoothing. |
| **`V7` (Dense Relational Loss)** | **Dense Relational Gram Matrix Alignment Loss** ($\mathcal{L}_{dense} + \mathcal{L}_{rel}$) | Preserves global intra-modality cell-cell relational geometry and prevents modality drift: <br> $\mathcal{L}_{dense} = \frac{1}{N}\left(\|\mathbf{z}_{joint} - \mathbf{z}_{RNA}'\|^2 + \|\mathbf{z}_{joint} - \mathbf{x}_{aux}\|^2\right) + \frac{0.1}{N^2}\|\mathbf{z}_{RNA}'(\mathbf{z}_{RNA}')^T - \mathbf{x}_{aux}\mathbf{x}_{aux}^T\|_F$ |
| **`V12` (Optimal Transport)** | **Sinkhorn-Wasserstein Relational Loss** ($\mathcal{L}_{OT}$) | Differentiable Entropic Optimal Transport aligning continuous geometric probability manifolds across RNA and Auxiliary modalities, resolving measurement scale mismatch and dropout. |
| **`V14` (Spatial Potts DEC)** | **Spatial Potts MRF-Regularized 2-Stage Consensus DEC** ($\mathcal{L}_{DEC-Potts}$) | Smooths target distribution $P$ using spatial Markov Random Field (Potts prior) neighborhood cluster consensus to eliminate salt-and-pepper domain misclassification artifacts: <br> $\mathcal{L}_{DEC-Potts} = \text{KL}(P \parallel Q) + \lambda_{Potts} \sum_{(i,j) \in \mathcal{E}} \|q_i - q_j\|_2^2$ |
| **Multi-Task Uncertainty** | **Kendall & Gal Homoscedastic Loss Balancing Engine** ($\mathcal{L}_{total}$) | Learns task-dependent log-variance parameters ($s_0, s_1, s_2, s_3, s_4 = \log \sigma_m^2$) during backpropagation for zero manual hyperparameter tuning: <br> $\mathcal{L}_{total} = \sum_{m=0}^4 \left( \frac{1}{2}\exp(-s_m)\mathcal{L}_m + \frac{1}{2}s_m \right)$ |

---

## 🌟 Architectural Dataflow of ARISE-Plus v2

```
                       [Spatial Multi-Omics Input]
                         /                     \
       (Transcriptomics: RNA)             (Epigenomics: ATAC / Proteomics: ADT)
               |                                           |
    [Multi-Order Topology (V9)]                    [Intersection Scaffold (V0)]
  - Cosine kNN Sim (G_sim)                       - G_comm' = G_sim' ∩ G_dist
  - Euclidean Dist (G_dist)                                |
  - 3-Node (M3) + 4-Node (M4) Motifs                       |
               |                                           |
      Dual GCN Encoders                            Scaffold GCN Encoder
       (x_sim, x_dist)                                  (x_aux)
               \                                           /
                ---> [RNA Latent Fusion: z_rna] <---------
                             |
         [Spatial Graph-Masked Cross-Attention (V11)]
         z_rna' = LayerNorm(z_rna + Softmax(QK^T/√d + M_spatial) · V)
                             |
         [Joint Multi-Modal Latent: z_joint]
                             |
    +------------------------+-----------------------------------+
    |                        |                                   |
[Multi-Head Decoders]  [Spatial Contrastive]   [Dense Gram Alignment (V7)] & [Sinkhorn OT (V12)]
(RNA, Aux, Joint Recon) (Physical Neighborhood) (||G_rna - G_aux||_F)        (<T, C>)
    |                        |                                   |
    +------------------------+-----------------------------------+
                             |
      [2-Stage Spatial Potts Consensus DEC (V14)]
       - Stage 1: Representation Pretraining
       - Stage 2: Spatial Potts MRF Target Distribution KL Fine-Tuning
                             |
     [Kendall & Gal Homoscedastic Multi-Task Loss Weighting]
      L = Σ 0.5 * exp(-s_m) * L_m + 0.5 * s_m (m = 0..4)
```

---

## 📁 Directory Files

- [**`ARISE_Plus_v2_Runner.py`**](file:///Users/imran/Developer/FYDP/forGit/arise-plus-v2/ARISE_Plus_v2_Runner.py): Standalone, CLI-driven executable training and evaluation script.
- [**`ARISE_Plus_v2_Colab.ipynb`**](file:///Users/imran/Developer/FYDP/forGit/arise-plus-v2/ARISE_Plus_v2_Colab.ipynb): Interactive Google Colab notebook for the main benchmark.
- [**`ARISE_Plus_v2_DEC_Ablation.py`**](file:///Users/imran/Developer/FYDP/forGit/arise-plus-v2/ARISE_Plus_v2_DEC_Ablation.py): Dedicated DEC component ablation suite tracking loss contributions and sub-embedding Silhouette score dynamics.
- [**`ARISE_Plus_v2_DEC_Ablation_Colab.ipynb`**](file:///Users/imran/Developer/FYDP/forGit/arise-plus-v2/ARISE_Plus_v2_DEC_Ablation_Colab.ipynb): Dedicated Google Colab notebook for running and visualizing DEC component ablations in the cloud.
- [**`README.md`**](file:///Users/imran/Developer/FYDP/forGit/arise-plus-v2/README.md): Comprehensive documentation and mathematical formulation.

---

## 🔬 DEC Component Ablation Studies

To investigate which component drives the Silhouette score improvement during Stage 2 (DEC fine-tuning):

### 1. Run DEC Ablation via CLI
```bash
# Full ablation suite on Dataset 0
python ARISE_Plus_v2_DEC_Ablation.py --dataset 0 --seeds 42 --out_dir results_dec_ablation

# Run selected ablations
python ARISE_Plus_v2_DEC_Ablation.py --dataset 0 --ablations Full_ARISE_v2,wo_Potts_MRF,wo_Sinkhorn_OT --seeds 42
```

### 2. Run in Google Colab
Open `ARISE_Plus_v2_DEC_Ablation_Colab.ipynb` on Google Colab to run the ablation matrix, display delta Silhouette rankings, and plot per-epoch trajectories.

---

## 💻 Local CLI Execution

### 1. Run All Benchmark Datasets (Lymph Node A1 & D1, Mouse Brain E11/E13/E15/E18)
```bash
python ARISE_Plus_v2_Runner.py --dataset all --seeds 42 1234 2024 --epochs 400 --out_dir results_arise_plus_v2
```

### 2. Run a Single Dataset (e.g. 10x Human Lymph Node A1)
```bash
python ARISE_Plus_v2_Runner.py --dataset 0 --seeds 42 --epochs 400 --out_dir results_arise_plus_v2
```

### 3. Customize Pre-training and DEC Fine-tuning Epochs
```bash
python ARISE_Plus_v2_Runner.py --dataset 0,1 --pretrain_epochs 250 --finetune_epochs 150 --lr 0.001
```

---

## ☁️ Google Colab Execution

1. Upload `ARISE_Plus_v2_Colab.ipynb` or `ARISE_Plus_v2_DEC_Ablation_Colab.ipynb` to [Google Colab](https://colab.research.google.com).
2. Set runtime type to **GPU** (`Runtime` -> `Change runtime type` -> `T4 GPU`).
3. Run all cells step-by-step.
