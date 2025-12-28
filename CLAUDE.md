# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GenRec-V1 ("Haitang, 海棠") is a research implementation for multimodal recommendation systems using generative approaches with flip-based interest generation. This is the official implementation accompanying the ACM MM 2025 paper "Flip is Better than Noise: Unbiased Interest Generation for Multimedia Recommendation".

**Paper**: [ACM MM 2025](https://dl.acm.org/doi/pdf/10.1145/3746027.3755743)

## Running the Model

### Basic Training Commands

```bash
# Train on TikTok dataset (default)
python Main.py --data tiktok

# Train on Baby dataset
python Main.py --data baby

# Train on Sports dataset
python Main.py --data sports
```

### GPU Configuration

GPU selection is controlled via `--gpu` parameter in Params.py (default: '1'). This is set at line 640 in Main.py via `os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu`.

```bash
# Use specific GPU
python Main.py --data tiktok --gpu 0
```

### Key Hyperparameters

Common parameters to adjust (see Params.py for full list):

```bash
python Main.py --data tiktok \
  --epoch 50 \
  --batch 1024 \
  --lr 1e-3 \
  --gen_topk 5 \
  --rebuild_k 1 \
  --sampling_steps 5 \
  --OpenInterestDebiase False
```

## Architecture Overview

### Three-Stage Training Pipeline

The model uses a unique three-stage training process within each epoch (see Main.py `trainEpoch`):

1. **Interest Generation via Diffusion** (lines 246-306)
   - Uses `FlipInterestDiffusion` to train denoising model
   - Processes user-item interaction graphs through diffusion process
   - Applies flip-based generation instead of noise-based

2. **UI Matrix Reconstruction** (lines 312-474)
   - Performs generative inference using trained diffusion model
   - Optionally applies `InterestDebiase` for unbiased interest filtering
   - Builds enhanced user-item interaction matrices per modality

3. **GCN Optimization** (lines 478-556)
   - Trains GCN model on reconstructed graphs
   - Computes BPR loss and contrastive learning losses
   - Updates user/item embeddings

### Core Components

**Main.py**: Training orchestration
- `Coach` class handles entire training loop
- `prepareModel()`: Initializes models and multimodal interest clustering (lines 88-173)
- `trainEpoch()`: Three-stage training process
- `testEpoch()`: Evaluation on test set

**Model.py**: Neural architectures
- `GCNModel`: Multi-modal graph convolutional network with separate encoders for image/text/audio features
- `FlipInterestDiffusion`: Implements flip-based diffusion for interest generation (not noise-based)
- `ModalDenoiseTransformer`: Transformer-based denoising model with multi-head attention

**interest_cluster.py**: Interest space construction
- `MultimodalCluster`: Creates modal-specific item clustering using K-means
  - Automatically determines optimal cluster numbers if `use_auto_optimal_k=True`
  - Default cluster numbers vary by dataset (TikTok: 18/59/46 for image/text/audio)
- `InterestDebiase`: Debiases generated interests using cluster-based sampling
  - Controlled by `--OpenInterestDebiase` flag
  - Uses `sample_ratio` to control random sampling proportion

**DataHandler.py**: Data loading pipeline
- Loads sparse user-item interaction matrices (trnMat.pkl, tstMat.pkl)
- Loads pre-extracted multimodal features (image_feat.npy, text_feat.npy, audio_feat.npy)
- Creates PyTorch dataloaders: `trnLoader`, `tstLoader`, `diffusionLoader`, `multimodalFeatureLoader`
- Constructs normalized adjacency matrices for GCN propagation

**Params.py**: Hyperparameter definitions
- All model/training hyperparameters defined here
- Default values optimized for TikTok dataset
- Key ablation flags: `OpenInterestDebiase`, `OpenUIG`, `OpenFlipGen`, `OpenTransformer`, `OpenMMCL`

### Data Flow

```
User-Item Matrix (sparse) → DiffusionLoader → FlipInterestDiffusion
                                                ↓
                                    Denoised UI Interactions
                                                ↓
                                    InterestDebiase (optional)
                                                ↓
                                    Reconstructed UI Matrices
                                                ↓
Multimodal Features → GCNModel ← Reconstructed Matrices
                         ↓
                User/Item Embeddings → BPR Loss + CL Loss
```

## Dataset Structure

Expected directory structure in `Datasets/`:

```
Datasets/
├── tiktok/
│   ├── trnMat.pkl          # Training user-item interactions (sparse matrix)
│   ├── tstMat.pkl          # Test user-item interactions
│   ├── image_feat.npy      # Image features (num_items × 128)
│   ├── text_feat.npy       # Text features (num_items × 768)
│   └── audio_feat.npy      # Audio features (num_items × 128)
├── baby/
│   ├── trnMat.pkl
│   ├── tstMat.pkl
│   ├── image_feat.npy.zip  # MUST unzip before use
│   └── text_feat.npy
└── sports/
    └── [Download from Google Drive - see README]
```

### Dataset Characteristics

From interest clustering analysis (Main.py lines 265-269):
- **TikTok**: interest_min=0, interest_mean=7, interest_max=603
- **Baby**: interest_min=3, interest_mean=7, interest_max=100
- **Sports**: interest_min=3, interest_mean=7, interest_max=237

These ranges inform optimal cluster numbers for `MultimodalCluster`.

## Important Implementation Details

### GPU Memory Management
- Models are loaded to GPU via `.cuda()` calls
- Detached embeddings used during diffusion training to prevent backprop through GCN
- User/item embeddings detached at line 278-279 before diffusion training

### Interest Clustering (Lines 97-144)
- Performed once during `prepareModel()`
- Creates item cluster labels for each modality
- Optimal cluster numbers dataset-specific:
  - TikTok: 18 (image), 59 (text), 46 (audio)
  - Baby: 6 (image), 11 (text)
  - Sports: 9 (image), 12 (text)
- Can auto-search with `--use_auto_optimal_k True` (uses elbow method)

### Item-Item Similarity Matrices (Lines 153-156)
- Built using `buildItem2ItemMatrix()` from modal features
- Creates KNN graphs with `--knn_k` neighbors (default: 5)
- Used in GCN propagation for content-based message passing

### Debiasing Module (Lines 398-436)
- Only active if `--OpenInterestDebiase True`
- Queries multimodal interest cluster space
- Filters generated interests that don't align with user's cluster distribution
- Controlled by `sample_ratio` hyperparameter

### Training Outputs
- TensorBoard logs saved to `runs/experiment/`
- Prints best epoch with Recall@20, NDCG@20, Precision@20
- Testing performed every `--tstEpoch` epochs (default: 1)

## Environment Requirements

```
Python >= 3.8
PyTorch >= 2.0
scipy == 1.9.1
scikit-learn >= 1.2.0
numpy >= 1.24.0
tensorboard (for logging)
```

## Common Issues

### Baby Dataset
Must unzip `Datasets/baby/image_feat.npy.zip` before training.

### Sports Dataset
Large file hosted on Google Drive - download separately from README link.

### Audio Features
Only TikTok dataset includes audio modality. Baby and Sports use image + text only.

### Memory Constraints
Reduce `--batch` size if encountering OOM errors. Default is 1024.

## Model Variants & Ablations

Control via boolean flags in Params.py:
- `--OpenInterestDebiase`: Enable/disable interest debiasing module
- `--OpenUIG`: Unbiased Interest Generation module
- `--OpenFlipGen`: Flip generation submodule
- `--OpenTransformer`: Use transformer vs basic denoiser
- `--OpenMMCL`: Multimodal contrastive learning

## Citation

```bibtex
@inproceedings{he2025flip,
  title={Flip is Better than Noise: Unbiased Interest Generation for Multimedia Recommendation},
  author={He, Yue and Xie, Jingxi and Li, Fengling and Zhu, Lei and Li, Jingjing},
  booktitle={Proceedings of the 33rd ACM International Conference on Multimedia},
  pages={6298--6306},
  year={2025}
}
```
