#!/bin/bash

# Check for clean flag as the very first argument
if [ "$1" = "clean" ]; then
  echo "Cleaning up previous build..."
  rm -rf build
  shift # Shift arguments so remaining flags match expected positions
fi

USE_CUDA=${1:-OFF}
CUDA_VERSION=${2:-12.8}


if [ -z "$3" ]; then
  TOTAL_CORES=$(nproc)
  NUM_THREADS=$(( TOTAL_CORES > 2 ? TOTAL_CORES - 2 : 1 ))
else
  NUM_THREADS=$3
fi

echo "Using CUDA: ${USE_CUDA}"
echo "CUDA Version: ${CUDA_VERSION}"
echo "Number of Threads: ${NUM_THREADS}"

# Validate NVCC compiler path only if CUDA is explicitly enabled
if [ "${USE_CUDA}" = "ON" ]; then
  NVCC_PATH="/usr/local/cuda-${CUDA_VERSION}/bin/nvcc"
  if [ ! -f "${NVCC_PATH}" ]; then
    echo "Error: CUDA compiler not found at ${NVCC_PATH}"
    exit 1
  fi
fi

if [ ! -d "build" ]; then
  mkdir build
fi

cmake -B build \
  -DUSE_CUDA="${USE_CUDA}" \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-"${CUDA_VERSION}"/bin/nvcc

cmake --build build -j"${NUM_THREADS}"
