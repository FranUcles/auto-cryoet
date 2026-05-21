# Installation

## Prerequisites

- [Conda](https://docs.conda.io/en/latest/) or [Mamba](https://mamba.readthedocs.io/en/latest/) installed on your system
- A CUDA-compatible GPU is required for executing TomoTwin

## Steps

### 1. Create the conda environment

```bash
mamba env create -n auto-cryoet -f https://raw.githubusercontent.com/MPI-Dortmund/tomotwin-cryoet/main/conda_env_tomotwin.yml
conda activate auto-cryoet
```

> If you don't have `mamba` installed, you can replace `mamba` with `conda`, though it will be slower.

### 2. Install TomoTwin

```bash
pip install tomotwin-cryoet
```

### 3. Install DisPerSE

```bash
conda install conda-forge::disperse
```

### 4. Install the tools

From the root of this repository, run:

```bash
pip install .
```

This will make all the tools available as commands in your terminal.