# Resource-Efficient CSI Prediction: A Gated Fusion and Factorized Projection Approach

[![arXiv](https://img.shields.io/badge/arXiv-2605.06578-b31b1b.svg)](https://arxiv.org/abs/2605.06578)
[![IEEE Xplore](https://img.shields.io/badge/IEEE%20Xplore-10.1109/LCOMM.2026.3691476-00629B.svg)](https://ieeexplore.ieee.org/document/11513393)
[![IEEE](https://img.shields.io/badge/IEEE-Communications%20Letters-blue.svg)](https://doi.org/10.1109/LCOMM.2026.3691476)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation of the paper:

> **Resource-Efficient CSI Prediction: A Gated Fusion and Factorized Projection Approach**
>
> [Mohammad Hussain](https://arxiv.org/search/eess?searchtype=author&query=Hussain,+M), [Maedeh Adibag](https://arxiv.org/search/eess?searchtype=author&query=Adibag,+M), [Dilara Gurer](https://arxiv.org/search/eess?searchtype=author&query=Gurer,+D), [Gokhan Kalem](https://arxiv.org/search/eess?searchtype=author&query=Kalem,+G), [Kerim Serin](https://arxiv.org/search/eess?searchtype=author&query=Serin,+K), [Sinem Coleri](https://arxiv.org/search/eess?searchtype=author&query=Coleri,+S)
>
> *Accepted for publication in **IEEE Communications Letters**, 2026*

### 📄 Paper Links
| | Link |
|---|---|
| **IEEE Xplore** | [https://ieeexplore.ieee.org/document/11513393](https://ieeexplore.ieee.org/document/11513393) |
| **arXiv** | [https://arxiv.org/abs/2605.06578](https://arxiv.org/abs/2605.06578) |
| **DOI** | [10.1109/LCOMM.2026.3691476](https://doi.org/10.1109/LCOMM.2026.3691476) |

---

## Abstract

Accurate Channel State Information (CSI) prediction is essential for dynamic multiple-input multiple-output (MIMO) systems but remains computationally demanding. This letter proposes a resource-efficient predictor that combines a gated recurrent unit (GRU) encoder with Luong attention, a bottleneck gated fusion module, and a Dimension-wise Separable Linear Head (DSLH). The gated fusion module integrates local recurrent features with global attention context, while the DSLH reduces the cost of the output mapping. Evaluated on 3GPP TR 38.901-compliant channels, the proposed model achieves an average NMSE of **−13.84 dB** with **26% fewer parameters** and approximately **2.3× higher inference throughput** than a dimension-matched LinFormer baseline. The proposed model is best suited to LOS and mixed-condition scenarios, offering a practical accuracy–efficiency trade-off for short-horizon CSI prediction at moderate sequence lengths.

---

## Architecture

The proposed **GRU-Attn-DSLH** model consists of four key components:

```
Input Sequence ──► GRU Encoder ──► Luong Attention ──► Bottleneck Gated Fusion ──► DSLH Head ──► Predicted CSI
  [B, Np, din]     (temporal)       (global context)    (adaptive integration)     (factorized)   [B, Nf, dout]
```

| Component | Description |
|---|---|
| **GRU Encoder** | Multi-layer GRU to capture temporal dynamics of the CSI sequence |
| **Luong-style Attention** | Dot-product attention to focus on the most relevant historical context |
| **Bottleneck Gated Fusion** | Adaptive gating mechanism that integrates local recurrent features with global attention context via a compressed bottleneck |
| **DSLH (Dimension-wise Separable Linear Head)** | Factorizes the output projection into separable time-mapping (Np→Nf) and channel-mapping (d_model→d_out) operations, significantly reducing parameters |

---

## Repository Structure

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── Code/
│   ├── streamlined_csi_prediction.ipynb   # Training notebook (all 4 models)
│   └── streamlined_inference.ipynb        # Inference & evaluation notebook
└── Dataset Generator/
    └── drive_dataset_generator_single_carrier.m  # MATLAB/QuaDRiGa dataset generator
```

### Code Overview

| File | Description |
|---|---|
| `streamlined_csi_prediction.ipynb` | End-to-end training pipeline: data loading, model definitions (proposed + 3 baselines), WMSE loss, cosine-warmup LR schedule, early stopping, and comparative training |
| `streamlined_inference.ipynb` | Multi-dataset evaluation with per-model NMSE, per-frame MSE, forecast plots, error histograms, heatmaps, leaderboard table, and efficiency benchmarks (params/FLOPs/throughput) |
| `drive_dataset_generator_single_carrier.m` | MATLAB script using [QuaDRiGa 2.8.1](https://quadriga-channel-model.de/) to generate 3GPP TR 38.901-compliant 2×2 MIMO channel datasets with variable-speed UE mobility profiles |

### Models Implemented

| Model | Type | Description |
|---|---|---|
| **GRU-Attn-DSLH** | Proposed | GRU encoder + Luong attention + Bottleneck gated fusion + DSLH |
| **Vanilla LSTM** | Baseline | Multi-layer LSTM + LayerNorm + DSLH |
| **Vanilla GRU** | Baseline | Multi-layer GRU + LayerNorm + DSLH |
| **LinFormer** | Baseline | TMLP-based encoder blocks + DSLH |

---

## Getting Started

### Prerequisites

- Python ≥ 3.8
- PyTorch ≥ 2.0
- MATLAB with [QuaDRiGa](https://quadriga-channel-model.de/) (for dataset generation only)

### Installation

```bash
git clone https://github.com/MHDoh/resource-efficient-csi-prediction.git
cd resource-efficient-csi-prediction
pip install -r requirements.txt
```

### Dataset Generation (Optional)

If you want to generate your own datasets using QuaDRiGa:

1. Install [QuaDRiGa 2.8.1-0](https://quadriga-channel-model.de/) in MATLAB.
2. Run the MATLAB script:
   ```matlab
   cd 'Dataset Generator'
   generate_quadriga_multiset()
   ```
   This produces 6 `.mat` files (HDF5 / `-v7.3` format):
   - `dataset_train_mixed.mat` — Training set (mixed scenarios, 3–60 km/h)
   - `dataset_test_uma_los.mat` — UMa LOS test (10–60 km/h)
   - `dataset_test_umi_nlos.mat` — UMi NLOS test (10–60 km/h)
   - `dataset_test_high_speed.mat` — High-speed test (80–120 km/h)
   - `dataset_test_pedestrian.mat` — Pedestrian test (3–5 km/h)
   - `dataset_test_general.mat` — General hold-out test (10–60 km/h)

### Training

1. Open `Code/streamlined_csi_prediction.ipynb` in Jupyter.
2. Update `cfg.mat_path` to point to your training `.mat` file.
3. Run all cells. Models are saved to `trained_streamlined/`.

### Inference & Evaluation

1. Open `Code/streamlined_inference.ipynb` in Jupyter.
2. Update the `DATASETS` dictionary with paths to your test `.mat` files.
3. Ensure trained checkpoints exist in `trained_streamlined/`.
4. Run all cells to generate NMSE tables, per-frame MSE curves, forecast visualizations, error heatmaps, and efficiency comparisons.

---

## Key Results

Evaluated on 3GPP TR 38.901-compliant channels (2×2 MIMO, 3.5 GHz):

| Model | Avg NMSE (dB) | Params | Relative Throughput |
|---|---|---|---|
| **GRU-Attn-DSLH (Proposed)** | **−13.84** | **~26% fewer** | **~2.3× faster** |
| LinFormer | Baseline | Baseline | Baseline |
| Vanilla LSTM | — | — | — |
| Vanilla GRU | — | — | — |

> *Refer to the paper for detailed per-scenario results.*

---

## Citation

If you use this code or find our work useful in your research, please cite our paper:

```bibtex
@article{hussain2026resource,
  title     = {Resource-Efficient {CSI} Prediction: A Gated Fusion and Factorized Projection Approach},
  author    = {Hussain, Mohammad and Adibag, Maedeh and Gurer, Dilara and Kalem, Gokhan and Serin, Kerim and Coleri, Sinem},
  journal   = {IEEE Communications Letters},
  year      = {2026},
  doi       = {10.1109/LCOMM.2026.3691476},
  note      = {arXiv:2605.06578}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Channel simulations generated using [QuaDRiGa](https://quadriga-channel-model.de/) (v2.8.1-0).
- 3GPP TR 38.901 channel model specifications.

---

## Contact

For questions or issues, please open a [GitHub Issue](../../issues) or contact the authors.
