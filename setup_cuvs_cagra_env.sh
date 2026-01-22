#!/usr/bin/env bash
set -euo pipefail

# ---- configuration (edit as needed) ----
ENV_NAME="${ENV_NAME:-cuvs-cagra}"
# Set CUDA_VERSION to the toolkit version supported by your driver (e.g. 11.8 or 12.1)
CUDA_VERSION="${CUDA_VERSION:-11.8}"

# ---- conda environment ----
conda create -y -n "$ENV_NAME" python=3.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Core scientific stack
conda install -y -c conda-forge numpy

# PyTorch with CUDA
conda install -y -c pytorch -c nvidia pytorch pytorch-cuda

# RAPIDS cuVS + CUDA meta package
conda install -y -c rapidsai -c conda-forge -c nvidia \
  cuvs "cuda-version=${CUDA_VERSION}"

# CuPy (CUDA-enabled)
conda install -y -c conda-forge cupy

# TensorFlow + ArrayRecord (for ArrayRecord data loading)
pip install --upgrade pip
pip install tensorflow array-record

# ---- runtime environment variables ----
# Update these to match your dataset location and schema if different.
export CLIC_DATA_DIR="${CLIC_DATA_DIR:-/workspace/data}"
export CLIC_ARRAY_RECORD_GLOB="${CLIC_ARRAY_RECORD_GLOB:-clic_edm_qq_pf-test.array_record-*}"
export CLIC_FEATURE_KEY="${CLIC_FEATURE_KEY:-X}"
export CLIC_FEATURE_WIDTH="${CLIC_FEATURE_WIDTH:-17}"

echo "Environment '$ENV_NAME' ready."
echo "Run: python Performance-Files/generate_gpu_cuvs_cagra.py"
