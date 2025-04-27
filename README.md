# Hybrid-sfqhull
High-performance CUDA implementations of convex hull algorithms with benchmarking.
🚀 Convex Hull Algorithms (CUDA Optimized)
High-performance CUDA implementations of classical and novel convex hull algorithms, benchmarked and compared for large-scale 2D point sets.

✨ Overview
This project implements and benchmarks various convex hull algorithms using CUDA C++, designed for parallel execution on GPUs.
The included algorithms range from classical approaches like QuickHull to novel hybrid optimizations like HeapHull and QuickHeapHull.

The goal is to evaluate and compare performance at scale, while exploring different strategies for parallelizing geometric algorithms.

🛠 Features
CUDA implementations of:

QuickHull

HeapHull

QuickHeapHull

Incremental Convex Hull

Parallel Scan Techniques

Synthetic data generation using Python (input_gen.ipynb)

Benchmarking across multiple datasets

Batch execution script (run_all.bat)

Clean modular structure for easy extension

📁 Project Structure
bash
Copy
Edit
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
🚀 Getting Started
📋 Requirements
CUDA Toolkit (v10.0 or higher recommended)

NVIDIA GPU with compute capability 5.0+

C++ Compiler

Python 3.x (for input generation, if needed)

numpy and matplotlib for input_gen.ipynb

🔧 Build Instructions
You can compile individual .cu files using nvcc:

bash
Copy
Edit
nvcc quickhull.cu -o quickhull
nvcc heaphull.cu -o heaphull
nvcc quickheaphull.cu -o quickheaphull
nvcc incremental.cu -o incremental
Or compile and run everything automatically:

bash
Copy
Edit
./run_all.bat
Make sure the .bat file paths match your local environment setup.

🏃‍♂️ Run Instructions
After building, simply execute:

bash
Copy
Edit
./quickhull
./heaphull
./quickheaphull
./incremental
Input data can be auto-generated or provided manually.

📊 Results
Benchmark results are recorded in results.txt.

Key findings include:


Algorithm	Runtime (ms)	Notes
QuickHull	XXX	Baseline
HeapHull	XXX	Improved load balancing
QuickHeapHull	XXX	Best hybrid performance
Incremental	XXX	High variance at scale
Exact runtime values are available inside the results.txt file.

📈 Future Work
Extend to 3D convex hulls

Experiment with shared memory optimizations

Adaptive load balancing during recursion

Visualization of the resulting convex hulls

Profiling with Nsight and optimizing warp divergence

🤝 Contributing
Pull requests are welcome!
If you have improvements, new algorithms, or suggestions, feel free to open an issue or a PR.

📜 License
Distributed under the MIT License.
See LICENSE for more information.

✨ Acknowledgements
Classical algorithms based on standard computational geometry literature

CUDA parallelization inspired by NVIDIA research papers

