# 🚀 Convex Hull Algorithms (CUDA Optimized)

> High-performance CUDA implementations of classical and novel convex hull algorithms, benchmarked and compared for large-scale 2D point sets.

## ✨ Overview

This project implements and benchmarks various **convex hull algorithms** using **CUDA C++**, designed for parallel execution on GPUs.  
The included algorithms range from classical approaches like **QuickHull** to novel hybrid optimizations like **HeapHull** and **QuickHeapHull**.

The goal is to evaluate and compare performance at scale, while exploring different strategies for parallelizing geometric algorithms.

---

## 🛠 Features

- CUDA implementations of:
  - **QuickHull**
  - **HeapHull**
  - **QuickHeapHull**
  - **Incremental Convex Hull**
  - **Parallel Scan Techniques**
- Synthetic data generation using Python (`input_gen.ipynb`)
- Benchmarking across multiple datasets
- Batch execution script (`run_all.bat`)
- Clean modular structure for easy extension

---

## 📁 Project Structure

```bash
pcp_final/
│
├── heaphull.cu         # CUDA implementation of HeapHull
├── incremental.cu      # Incremental convex hull algorithm
├── quickhull.cu        # Standard QuickHull algorithm
├── quickheaphull.cu    # Hybrid QuickHeapHull approach
├── scan.cu             # Parallel scan (prefix sum) utilities
│
├── input_gen.ipynb     # Jupyter Notebook to generate synthetic inputs
├── run_all.bat         # Batch script to compile & run all algorithms
│
├── results.txt         # Benchmark results (runtime comparisons)
└── readme.txt          # (Legacy README - superseded by this file)

## 🚀 Getting Started

### 📋 Requirements

- CUDA Toolkit (v10.0 or higher recommended)
- NVIDIA GPU with compute capability 5.0+
- C++ Compiler
- Python 3.x (for input generation, if needed)
- `numpy` and `matplotlib` for `input_gen.ipynb`

### 🔧 Build Instructions

You  Contributing
Pull requests are welcome!
If you have improvements, new algorithms, or suggestions, feel free to open an issue or a PR.

📜 License
Distributed under the MIT License.
See LICENSE for more information.

✨ Acknowledgements
Classical algorithms based on standard computational geometry literature

CUDA parallelization inspired by NVIDIA research paperscan compile individual `.cu` files using `nvcc`:

```bash
nvcc quickhull.cu -o quickhull
nvcc heaphull.cu -o heaphull
nvcc quickheaphull.cu -o quickheaphull
nvcc incremental.cu -o incremental

Or compile and run everything automatically:
./run_all.bat
(Make sure the .bat file paths match your local environment setup.)

Results
Benchmark results are recorded in results.txt.p


