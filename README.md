<div align="center">

<img src="https://raw.githubusercontent.com/Samsomyajit/pibert/main/SPFG-removebg-preview.png" alt="PIBERT Logo" width="180"/>

# PIBERT: Physics-Informed BERT-style Transformer

[![PyPI version](https://img.shields.io/pypi/v/pibert?style=flat-square)](https://pypi.org/project/pibert/)
[![Python Version](https://img.shields.io/pypi/pyversions/pibert?style=flat-square)](https://pypi.org/project/pibert/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=flat-square)](https://opensource.org/licenses/Apache-2.0)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.jcp.2024.xxxx-blue?style=flat-square)](https://doi.org/10.1016/j.jcp.2024.xxxx)
[![SSRN](https://img.shields.io/badge/SSRN-6203487-2C5CC5.svg?style=flat-square&logo=bookstack&logoColor=white)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6203487)

![PIBERT Visual Abstract](https://github.com/Samsomyajit/pibert/blob/main/FIGURE/pibert_methodology_figure%20(1).png)

*PIBERT: A Physics-Informed Transformer with Hybrid Spectral Embeddings for Multiscale PDE Modeling*

</div>

## Artifact Bundles

The latest final figures, supplementary figures, and lightweight RealPDEBench result summaries are tracked in this branch. See [ARTIFACTS.md](ARTIFACTS.md) for the published bundle layout and for the list of manuscript files and oversized raw artifacts that are intentionally excluded from git.

## Introduction

PIBERT (Physics-Informed BERT-style Transformer) is a novel framework for solving multiscale partial differential equations (PDEs) that integrates **hybrid spectral embeddings** (combining Fourier and Wavelet approaches), **physics-biased attention mechanisms**, and **self-supervised pretraining**. 

Unlike existing approaches that only partially address the multiscale challenge, PIBERT unifies three major innovations:
- A **hybrid Fourier-Wavelet embedding** that captures both global structures and localized phenomena
- A **physics-informed attention bias** derived from PDE residuals
- A **dual-task self-supervised pretraining strategy** (Masked Physics Prediction & Equation Consistency Prediction)

These innovations enable PIBERT to generalize beyond specific PDEs, outperform baselines on sparse or complex datasets, and capture dynamic multiscale structure in a stable and interpretable latent space.

## Key Features

- **Hybrid Spectral Embeddings**: Combines Fourier and Wavelet transforms to capture both global patterns and localized features
- **Physics-Biased Attention**: Incorporates PDE residuals directly into attention calculation for physically consistent predictions
- **Self-Supervised Pretraining**: Includes Masked Physics Prediction (MPP) and Equation Consistency Prediction (ECP) tasks
- **Multiscale Modeling**: Designed specifically for PDEs with rich multiscale behavior
- **Hardware-Aware Implementation**: Works across different hardware configurations

## Hardware Requirements

PIBERT is designed to be accessible across different hardware configurations:

| Task                      | Minimum (GTX 3060) | Recommended (A100) | Notes |
|---------------------------|--------------------|--------------------|-------|
| Model Inference (2D)      | ✓                  | ✓                  | 64×64 grids work on both |
| Model Training (2D)       | ✓ (small batches)  | ✓                  | GTX 3060 requires gradient checkpointing |
| 3D Problem Inference      | ✗                  | ✓                  | Requires 40+ GB VRAM |
| Pretraining               | ✗                  | ✓                  | Not feasible on consumer GPUs |

## Installation

```bash
# Basic installation
pip install pibert

# For development with testing and documentation tools
pip install pibert[dev]

# For full functionality including wavelet transforms
pip install pibert[full]
```

## Quick Start

Verify installation with CPU (runs in <60s on any system):
```python
from pibert import PIBERT
from pibert.utils import load_dataset

# Load a small sample dataset
dataset = load_dataset("reaction_diffusion")

# Initialize a small model
model = PIBERT(
    input_dim=1,
    hidden_dim=64,
    num_layers=2,
    num_heads=4
)

# Perform prediction
pred = model.predict(dataset["test"]["x"][:1], dataset["test"]["coords"][:1])

print(f"Prediction shape: {pred.shape}")
```

For more examples, see the [examples directory](examples/).

## Performance Comparison

PIBERT demonstrates state-of-the-art performance across multiple benchmarks:

### 1D Reaction Equation
| Model | Relative L1 | Relative L2 | MAE |
|-------|-------------|-------------|-----|
| PINN | 0.0651 | 0.0803 | 0.0581 |
| FNO | 0.0123 | 0.0150 | 0.0100 |
| Transformer | 0.0225 | 0.0243 | 0.0200 |
| PINNsFormer | 0.0065 | 0.0078 | 0.0060 |
| **PIBERT** | **0.0061** | **0.0074** | **0.0056** |

### CFDBench (Cavity Flow)
| Model | MSE(u) | MSE(v) | MSE(p) |
|-------|--------|--------|--------|
| PINNs | 0.0500 | 0.0300 | 0.01500 |
| Spectral PINN | 0.0200 | 0.0045 | 0.00085 |
| FNO | 0.0113 | 0.0012 | 0.00021 |
| PINNsFormer | 0.0065 | 0.0007 | 0.00003 |
| **PIBERT(Lite)** | **0.0103** | **0.0011** | **0.000046** |

## Ablation Study Results

The ablation study confirms the importance of each component:

| Model Variant | MSE (Test) | NMSE (Test) |
|---------------|------------|-------------|
| PIBERT (Full) | 0.4975 | 1.3409 |
| Fourier-only | 1.6520 | 12.4010 |
| Wavelet-only | **0.4123** | **1.1021** |
| Standard-attention | 1.3201 | 9.8760 |
| FNO | 1.8099 | 13.5830 |
| UNet | 3.7006 | 29.2627 |

Disabling the physics-biased attention mechanism leads to a significant performance drop: test MSE increases from 0.4975 to 1.3201, and NMSE jumps from 1.34 to 9.88.

## PIBERT on EAGLE and CFDBench
### EAGLE
![EAGLE](https://github.com/Samsomyajit/pibert/blob/main/eagle.png)

### Cylinder Wake
![CFDBench Cylinder-wake](https://github.com/Samsomyajit/pibert/blob/main/cylinder.png)

### CFD Bench
![CFDBench Cylinder-wake](https://github.com/Samsomyajit/pibert/blob/main/fig_grid_clean.png)

---

## RealPDEBench Results

PIBERT was evaluated on the **RealPDEBench** benchmark suite covering two challenging real-world fluid dynamics datasets: Cylinder Wake and Fluid-Structure Interaction (FSI).

### Cylinder Wake (Real)

All-component metrics (u + v combined):

| Model | MSE | NMSE | LMAE | LPCC | R² |
|-------|-----|------|------|------|----|
| PINN | 0.4269 | 0.3994 | 0.2929 | 0.7749 | 0.6005 |
| DeepONet2d | 0.2122 | 0.1985 | 0.2619 | 0.8966 | 0.8015 |
| PITT | 0.1254 | 0.1173 | 0.1910 | 0.9412 | 0.8827 |
| FourierFlow | 0.0634 | 0.0593 | 0.1180 | 0.9699 | 0.9407 |
| FNO2d | 0.0642 | 0.0600 | 0.1322 | 0.9696 | 0.9400 |
| **PIBERT** | **0.0628** | **0.0587** | **0.1145** | **0.9702** | **0.9412** |

#### Model Overview — Cylinder Real
![Cylinder Real Overview](https://github.com/Samsomyajit/pibert/blob/main/FIGURE/main_cylinder_overview.png)

#### Predicted Fields — Cylinder Real
![Cylinder Real Fields](https://github.com/Samsomyajit/pibert/blob/main/FIGURE/main_cylinder_pibert_fields.png)

#### Metrics Comparison — Cylinder Real
![Cylinder Real Metrics](https://github.com/Samsomyajit/pibert/blob/main/FIGURE/main_cylinder_metrics.png)

#### Multiscale Analysis — Cylinder Real
![Cylinder Real Multiscale](https://github.com/Samsomyajit/pibert/blob/main/FIGURE/main_cylinder_multiscale.png)

---

### Fluid-Structure Interaction (FSI Real)

All-component metrics (u + v combined):

| Model | MSE | NMSE | LMAE | LPCC | R² |
|-------|-----|------|------|------|----|
| PITT | 0.03580 | 0.04690 | 0.10134 | 0.9761 | 0.9528 |
| DeepONet2d | 0.08029 | 0.10518 | 0.17215 | 0.9456 | 0.8941 |
| PINN | 0.000442 | 0.000580 | 0.011629 | 0.9997 | 0.9994 |
| FNO2d | 0.001716 | 0.002248 | 0.027409 | 0.9989 | 0.9977 |
| FourierFlow | 0.000307 | 0.000402 | 0.010626 | 0.9998 | 0.9996 |
| **PIBERT** | **0.000206** | **0.000270** | **0.008640** | **0.9999** | **0.9997** |

#### Model Overview — FSI Real
![FSI Real Overview](https://github.com/Samsomyajit/pibert/blob/main/FIGURE/main_fsi_overview.png)

#### Predicted Fields — FSI Real
![FSI Real Fields](https://github.com/Samsomyajit/pibert/blob/main/FIGURE/main_fsi_pibert_fields.png)

#### Metrics Comparison — FSI Real
![FSI Real Metrics](https://github.com/Samsomyajit/pibert/blob/main/FIGURE/main_fsi_metrics.png)

#### Multiscale Analysis — FSI Real
![FSI Real Multiscale](https://github.com/Samsomyajit/pibert/blob/main/FIGURE/main_fsi_multiscale.png)

---

### Training Convergence
![Training Convergence](https://github.com/Samsomyajit/pibert/blob/main/FIGURE/main_training_convergence.png)

---

## Reproducibility

All results in the paper can be reproduced using the provided code. The ablation studies were verified on a GTX 3060 (12GB VRAM), while the full-scale experiments used A100 GPUs. We provide configuration files for both hardware setups.

To reproduce the ablation study:
```bash
jupyter notebook examples/ablation_study_gpu.ipynb
```

## Citing PIBERT

If you find PIBERT useful in your research, please cite our paper: [!Link](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6203487)

```bibtex
@article{chakraborty6203487pibert,
  title={PIBERT: A Physics-Informed Bi-directional Hybrid Spectral Transformer for Multiscale CFD Surrogate Modeling},
  author={Chakraborty, Somyajit and Pan, Ming and Chen, Xizhong},
  journal={Available at SSRN 6203487}
}
```

## License

Distributed under the Apache 2.0 License. See [LICENSE](LICENSE) for more information.

## Support

For support and questions, please open an issue on GitHub or contact the authors:
- Somyajit Chakraborty: [chksomyajit@sjtu.edu.cn](mailto:chksomyajit@sjtu.edu.cn)
- Pan Ming: [panming@sjtu.edu.cn](mailto:panming@sjtu.edu.cn)
- Chen Xizhong: [chenxizh@sjtu.edu.cn](mailto:chenxizh@sjtu.edu.cn)

---

*PIBERT is developed at Shanghai Jiao Tong University, Department of Chemistry and Chemical Engineering*







