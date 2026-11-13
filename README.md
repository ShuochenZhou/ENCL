# ENCL

Code for **[ENCL: Enabling Efficient Neuromorphic Continual Learning with Local Learning Rule and Sub-Network Freezing Strategy](https://arxiv.org/abs/2610.31927)**.

Rachmad Vidya Wicaksana Putra, Shuochen Zhou, Hassan Raza, and Muhammad Shafique.

ENCL combines a local learning rule with task-specific subnetwork freezing for task-incremental learning in spiking neural networks. This repository provides ENCL and three comparison methods on Split MNIST, Split Fashion-MNIST, Split CIFAR-10, and Split CIFAR-100.

## Setup

Use Linux, Python 3.10, and an NVIDIA GPU with CUDA and NVML energy-counter support. Run commands from the repository root in a dedicated environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install numpy==1.26.4 nvidia-ml-py
```

The alternative CUDA 12.1 environment uses PyTorch 2.2.2 and torchvision 0.17.2 from the `https://download.pytorch.org/whl/cu121` index. Dataset downloads and Split CIFAR-100 cache preparation happen automatically on first use.

## Run an experiment

```bash
python experiment_scripts/run_experiment.py split_mnist/encl --gpu 0 --data-root ../dataset
```

The experiment identifier is `<dataset>/<method>`:

| Dataset | Identifier | Tasks | Classes per task |
| --- | --- | ---: | ---: |
| Split MNIST | `split_mnist` | 5 | 2 |
| Split Fashion-MNIST | `split_fmnist` | 5 | 2 |
| Split CIFAR-10 | `split_cifar10` | 5 | 2 |
| Split CIFAR-100 | `split_cifar100` | 10 | 10 |

| Method | Identifier |
| --- | --- |
| ENCL | `encl` |
| BPTT without continual learning | `bptt_noncl` |
| TESS without continual learning | `tess_noncl` |
| Memory replay | `mr` |

`--gpu` selects the physical GPU index; the launcher sets CUDA visibility. Use a GPU without other compute workloads when collecting power and energy measurements. Add `--print-only` to inspect a command without downloading data or training.

## Run all 16 experiments

```bash
for dataset in split_mnist split_fmnist split_cifar10 split_cifar100; do
  for method in encl bptt_noncl tess_noncl mr; do
    python experiment_scripts/run_experiment.py "$dataset/$method" --gpu 0 --data-root ../dataset || exit 1
  done
done
```

Seeds, task splits, epochs, timesteps, and learning rates are defined in [`source_code/config.py`](source_code/config.py). Each run creates a timestamped directory under `outputs/`; use `--output PATH` to choose an empty directory. Generated logs, measurements, and other run artifacts remain local and are excluded from version control.

## Repository structure

```text
experiment_scripts/run_experiment.py   Experiment launcher
source_code/config.py                 Dataset and method configurations
source_code/methods/                   ENCL and comparison methods
source_code/models/                    Spiking networks and learning layers
source_code/data/                      Task splits and dataset preparation
source_code/common/                    Experiment setup and resource measurement
```

## Citation

```bibtex
@misc{putra2026encl,
  title={ENCL: Enabling Efficient Neuromorphic Continual Learning with Local Learning Rule and Sub-Network Freezing Strategy},
  author={Rachmad Vidya Wicaksana Putra and Shuochen Zhou and Hassan Raza and Muhammad Shafique},
  year={2026},
  eprint={2610.31927},
  archivePrefix={arXiv},
  primaryClass={cs.NE},
  url={https://arxiv.org/abs/2610.31927}
}
```
