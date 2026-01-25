# FastGraphCompute Performance Analysis & Visualization

This repository contains benchmark runners and plotting tools for comparing
[FastGraphCompute](https://github.com/jkiesele/FastGraphCompute/) (FGC) against
state-of-the-art nearest-neighbor search algorithms on CPU and GPU.

## Overview

The current workflow focuses on GPU results (FAISS-GPU, cuVS CAGRA, GGNN, FGC)
and maintains CPU runners (FAISS, ScaNN, HNSWLIB, Annoy, FGC). The plotting
pipeline consumes standardized per-algorithm CSVs from `gpu-performance-data/`
and produces publication-ready figures in `Plots/`.

## Repository Structure

```
.
├── process_gpu_csvs.py            # GPU plotting pipeline
├── generate_runs_files.py         # Build CPU/GPU run queues
├── generate_cpu_performance_data.py
├── generate_gpu_performance_data.py
├── generate_gpu_faiss.py
├── generate_gpu_ggnn.py
├── generate_gpu_cuvs_cagra.py
├── generate_gpu_fgc.py
├── generate_faiss_performance.py  # Legacy CPU runners
├── generate_hnswlib_performance.py
├── generate_annoy_performance.py
├── generate_ggnn_performance.py
├── check_cagra_data.py            # cuVS CAGRA sanity checks
├── gpu-performance-data/          # Cleaned per-algo CSVs (plot input)
├── Plots/                         # Generated figures (PNG)
├── cpu-runs.txt                   # CPU run queue
├── gpu-runs-*.txt                 # GPU per-algo run queues
├── gpu_*_performance.csv          # Raw GPU benchmark output
├── cpu_performance.csv            # Raw CPU benchmark output
├── vector-analysis.ipynb
├── data-analysis.py
└── Old-Files/                     # Legacy scripts/data
```

## Installation

### Prerequisites

- Python 3.8+
- pip

### Plotting Dependencies

```bash
pip install pandas numpy plotly kaleido
```

### Benchmark Dependencies

Benchmark scripts require algorithm-specific libraries and data loaders,
including (but not limited to) `torch`, `fastgraphcompute`, `faiss`,
`hnswlib`, `scann`, `annoy`, `cuvs`, `tensorflow`, `array_record`, and `h5py`.

## Usage

### 1) Generate Run Queues

```bash
python generate_runs_files.py
```

This writes `all-runs.txt`, `cpu-runs.txt`, and per-GPU queues like
`gpu-runs-faiss_gpu.txt`.

### 2) Run Benchmarks

CPU (consumes `cpu-runs.txt`, writes `cpu_performance.csv`):

```bash
python generate_cpu_performance_data.py
```

GPU (per-algorithm, consumes `gpu-runs-*.txt`, writes `gpu_*_performance.csv`):

```bash
python generate_gpu_faiss.py
python generate_gpu_ggnn.py
python generate_gpu_cuvs_cagra.py
python generate_gpu_fgc.py
```

Optional combined GPU runner (consumes `gpu-runs.txt`):

```bash
python generate_gpu_performance_data.py
```

### 3) Prepare Plot Inputs

Ensure standardized GPU CSVs live in `gpu-performance-data/`:

```
gpu-performance-data/
├── faiss-gpu.csv
├── cuvs-cagra.csv
├── ggnn.csv
└── fgc-gpu.csv
```

See `process_gpu_csvs.py` for expected columns and status filtering logic.

### 4) Generate Plots

```bash
python process_gpu_csvs.py
```

## Generated Visualizations

Outputs are saved in `Plots/` and include:

- `gpu_fgc_speedup_d3_all_algorithms.png`
- `gpu_fgc_speedup_d5_all_algorithms.png`
- `gpu_fgc_k_comparison_1M_d2-10_all_algorithms.png`
- `gpu_fgc_dimensional_scaling_1M_k40_all_algorithms.png`
- `gpu_fgc_speedup_d3_vs_d5_k40_all_algorithms.png`
- `gpu_fgc_speedup_d3_k40_log_y.png`

## Data Environment

Most benchmark runners read CLIC data from environment variables:

- `CLIC_DATA_DIR`
- `CLIC_ARRAY_RECORD_GLOB`
- `CLIC_FEATURE_KEY`
- `CLIC_FEATURE_WIDTH`
- `CLIC_HDF5_PATH`
- `CLIC_HDF5_DATASET`

Refer to `generate_cpu_performance_data.py` and the GPU runners for defaults.

## Legacy Code

Older CSV processing and plotting utilities live in `Old-Files/` for reference.

## Related Repository

The benchmark data is intended to align with:
**[FastGraphCompute](https://github.com/jkiesele/FastGraphCompute/)**.

## License

This repository is part of the FastGraphCompute research project. Please refer
to the main repository for licensing information.

## Citation

If you use these visualization tools in your research, please cite the
FastGraphCompute paper (forthcoming) and this repository.

