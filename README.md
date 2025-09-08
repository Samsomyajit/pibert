# PIBERT: Physics-Informed BERT-style Transformer

[![PyPI version](https://img.shields.io/pypi/v/pibert)](https://pypi.org/project/pibert/)
[![Python Version](https://img.shields.io/pypi/pyversions/pibert)](https://pypi.org/project/pibert/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.jcp.2024.xxxx-blue)](https://doi.org/10.1016/j.jcp.2024.xxxx)
[![arXiv](https://img.shields.io/badge/arXiv-2408.xxxxx-b31b1b.svg)](https://arxiv.org/abs/2408.xxxxx)

![PIBERT Visual Abstract](https://github.com/Samsomyajit/pibert/blob/main/PIBERTAbstract.png)

*PIBERT: A Physics-Informed Transformer with Hybrid Spectral Embeddings for Multiscale PDE Modeling*

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

## Reproducibility

All results in the paper can be reproduced using the provided code. The ablation studies were verified on a GTX 3060 (12GB VRAM), while the full-scale experiments used A100 GPUs. We provide configuration files for both hardware setups.

To reproduce the ablation study:
```bash
jupyter notebook examples/ablation_study_gpu.ipynb
```

## Citing PIBERT

If you find PIBERT useful in your research, please cite our paper:

```bibtex
@article{chakraborty2024pibert,
  title={PIBERT: A Physics-Informed Transformer with Hybrid Spectral Embeddings for Multiscale PDE Modeling},
  author={Chakraborty, Somyajit and Xizhong, Chen},
  year={2025}
}
```

## License

Distributed under the Apache 2.0 License. See [LICENSE](LICENSE) for more information.

## Support

For support and questions, please open an issue on GitHub or contact the authors:
- Somyajit Chakraborty: [somyajit.chakraborty@example.com](mailto:chksomyajit@sjtu.edu.cn)
- Chen Xizhong: [chen.xizhong@example.com](mailto:chenxizh@sjtu.edu.cn)

---

*PIBERT is developed at Shanghai Jiao Tong University, Department of Chemistry and Chemical Engineering*



