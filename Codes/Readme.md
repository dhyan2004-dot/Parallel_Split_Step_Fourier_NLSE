# Parallel Split-Step Fourier Method — 1D NLSE Solver

Course project for **ID5130 — Parallel Scientific Computing, IIT Madras**.

## Project Structure

```text
Codes/
├── Benchmark/
│   ├── serial_benchmark.cpp
│   ├── openmp_benchmark.cpp
│   ├── openacc_benchmark.cpp
│   ├── fft_serial_benchmark.cpp
│   ├── cufft_benchmark.cpp
│   ├── plot_benchmarks.py
│   └── plot_fft_benchmarks.py
│
├── Validation/
│   ├── serial_val.cpp
│   ├── openmp_val.cpp
│   ├── openacc_val.cpp
│   ├── fft_val.cpp
│   ├── cufft_val.cpp
│   └── compare_solutions.py
│
├── nlse_dynamics.cpp
└── plot_fiber_dynamics.py
```

## Dependencies

The project requires:

* **g++** — Serial and OpenMP implementations
* **nvc++** — NVIDIA HPC SDK compiler for OpenACC and cuFFT implementations
* **FFTW3** — FFT-based CPU implementation
* **cuFFT** — NVIDIA GPU FFT implementation
* **Python 3**

  * NumPy
  * Pandas
  * Matplotlib

## Benchmarking

The `Benchmark` folder contains automated benchmarking scripts. The scripts compile the required C++ implementations, execute them, collect the results, and generate plots automatically.

### Naive DFT Benchmarks

Benchmarks the Serial, OpenMP, and OpenACC implementations:

```bash
cd Codes/Benchmark
python3 plot_benchmarks.py
```

Output:

```text
benchmark_results.png
```

The benchmark sweeps grid sizes from **N = 512 to 8192**, running 100 SSFM steps for each size and repeating each measurement three times.

### FFT Library Benchmarks

Compares the FFTW3 CPU implementation against the cuFFT GPU implementation:

```bash
cd Codes/Benchmark
python3 plot_fft_benchmarks.py
```

Output:

```text
fft_benchmark_results.png
```

Grid sizes range from **N = 512 to 65536**, with 100 SSFM steps per size and three repetitions.

Raw benchmark output is also saved as:

```text
<name>_out.txt
```

## Validation

The `Validation` folder contains implementations used to verify the correctness of the parallel solvers against the serial reference solution.

The validation programs must first be compiled and executed manually.

### 1. Compile

```bash
cd Codes/Validation

g++ -O3 serial_val.cpp -o serial_val

g++ -O3 -fopenmp openmp_val.cpp -o openmp_val

nvc++ -acc -fast openacc_val.cpp -o openacc_val

g++ -O3 fft_val.cpp -o fft_val -lfftw3

nvc++ -acc -fast -cudalib=cufft cufft_val.cpp -o cufft_val
```

### 2. Run

```bash
./serial_val
./openmp_val
./openacc_val
./fft_val
./cufft_val
```

The programs generate:

```text
serial_out.csv
openmp_out.csv
openacc_out.csv
fft_out.csv
cufft_out.csv
```

### 3. Compare Solutions

```bash
python3 compare_solutions.py
```

The script computes the **L∞ norm error** of each parallel implementation relative to the serial solution and prints a PASS/FAIL summary.

It also generates:

```text
solution_comparison.png
```

> **Note:** FFTW3 and cuFFT comparisons are commented out by default in `compare_solutions.py`. They can be enabled if required.

## NLSE Fiber Dynamics

The root-level `nlse_dynamics.cpp` simulates the propagation of a Gaussian pulse through a strongly nonlinear fiber using the **Split-Step Fourier Method (SSFM)**.

The simulation uses:

```text
γ = 5.0
```

and FFTW3 for the Fourier transforms.

The complete spatio-temporal evolution is written to:

```text
wavefunction_dynamics.csv
```

### 1. Compile

```bash
g++ -O3 nlse_dynamics.cpp -o nlse_dynamics -lfftw3
```

### 2. Run

```bash
./nlse_dynamics
```

### 3. Generate Plots

```bash
python3 plot_fiber_dynamics.py
```

The plotting script reads `wavefunction_dynamics.csv` and generates visualizations of the pulse dynamics during propagation.

## Quick Start

For benchmarking:

```bash
cd Codes/Benchmark
python3 plot_benchmarks.py
python3 plot_fft_benchmarks.py
```

For validation:

```bash
cd Codes/Validation

g++ -O3 serial_val.cpp -o serial_val
g++ -O3 -fopenmp openmp_val.cpp -o openmp_val
nvc++ -acc -fast openacc_val.cpp -o openacc_val
g++ -O3 fft_val.cpp -o fft_val -lfftw3
nvc++ -acc -fast -cudalib=cufft cufft_val.cpp -o cufft_val

./serial_val
./openmp_val
./openacc_val
./fft_val
./cufft_val

python3 compare_solutions.py
```

For the fiber dynamics simulation:

```bash
g++ -O3 nlse_dynamics.cpp -o nlse_dynamics -lfftw3
./nlse_dynamics
python3 plot_fiber_dynamics.py
```

## Implementations

The project compares multiple approaches to solving the 1D NLSE using the Split-Step Fourier Method:

| Implementation | Parallelization / Library     |
| -------------- | ----------------------------- |
| Serial         | Standard C++                  |
| OpenMP         | CPU shared-memory parallelism |
| OpenACC        | GPU acceleration              |
| FFTW3          | CPU FFT library               |
| cuFFT          | NVIDIA GPU FFT library        |

The project therefore evaluates both **parallelization strategies** and the performance advantage of replacing the naive DFT with optimized FFT libraries.
