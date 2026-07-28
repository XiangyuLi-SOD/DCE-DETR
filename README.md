<div align="center">

# DCE-DETR: Dynamic Multi-Scale Mixture-of-Experts with Cross-Scale Feature Enhancement for UAV Small Object Detection

[![Python](https://img.shields.io/badge/Python-3.11.9-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)
[![Ubuntu](https://img.shields.io/badge/Ubuntu-24.04-E95420?logo=ubuntu&logoColor=white)](https://ubuntu.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<p align="center">
  <strong>Official Implementation</strong> 🚀
</p>

</div>

---

## 📋 Environment

| Component | Version |
|:----------|:--------|
| **OS** | Ubuntu 24.04 LTS |
| **CUDA** | 12.8 |
| **Python** | 3.11.9 |
| **PyTorch** | 2.9.1 |

---

## ⚙️ Setup

### 1. Create Conda Environment

```bash
conda create -n dce-detr python=3.11.9 -y
conda activate dce-detr
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> 💡 **Tip:** Ensure your CUDA driver is compatible with CUDA 12.8 before installing PyTorch. You can verify with:
> ```bash
> nvidia-smi
> ```

---

## 📂 Project Structure

```
DCE-DETR/
├── configs/           # Model configurations
├── datasets/          # Dataset loading & preprocessing
├── models/            # DCE-DETR architecture
├── utils/             # Helper utilities
├── tools/             # Training & evaluation scripts
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

---

## 🚀 Quick Start

```bash
# Activate environment
conda activate dce-detr

# Training
python tools/train.py --config configs/dce_detr_r50.yaml

# Evaluation
python tools/eval.py --config configs/dce_detr_r50.yaml --weights checkpoints/best.pth
```

---

## 📖 Citation

If you find this work helpful, please consider citing:

```bibtex
@article{dce-detr2026,
  title={DCE-DETR: Dynamic Multi-Scale Mixture-of-Experts with Cross-Scale Feature Enhancement for UAV Small Object Detection},
  journal={},
  year={2026}
}
```

---

<div align="center">
  <p>Made with ❤️ for UAV Small Object Detection</p>
</div>
