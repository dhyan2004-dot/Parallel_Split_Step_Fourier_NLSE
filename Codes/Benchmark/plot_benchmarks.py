import subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14, 'axes.labelsize': 14, 'xtick.labelsize': 12, 'ytick.labelsize': 12, 'axes.titlesize': 16, 'legend.fontsize': 12})

def run_cmd(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

import os

def run_and_parse(exe_name):
    print(f"\nExecuting {exe_name}")
    os.system(f"./{exe_name} > {exe_name}_out.txt")
    
    with open(f"{exe_name}_out.txt", "r") as f:
        output = f.read()
    print(output)
    
    times = {}
    for line in output.split('\n'):
        parts = line.split()
        if len(parts) == 6 and parts[0].isdigit():
            try:
                N = int(parts[0])
                avg_time = float(parts[2])
                times[N] = avg_time
            except ValueError:
                pass
    return times

def main():
    print("Compiling benchmarks")
    run_cmd("g++ -O3 serial_benchmark.cpp -o serial_benchmark")
    run_cmd("g++ -O3 -fopenmp openmp_benchmark.cpp -o openmp_benchmark")
    run_cmd("nvc++ -acc -fast -Minfo=accel openacc_benchmark.cpp -o openacc_benchmark")
    
    times_serial = run_and_parse("serial_benchmark")
    times_omp = run_and_parse("openmp_benchmark")
    times_acc = run_and_parse("openacc_benchmark")
    
    plt.figure(figsize=(10, 6))
    
    if times_serial:
        N_serial = sorted(times_serial.keys())
        T_serial = [times_serial[n] for n in N_serial]
        plt.plot(N_serial, T_serial, 'o-', linewidth=2, label='Serial')
        
    if times_omp:
        N_omp = sorted(times_omp.keys())
        T_omp = [times_omp[n] for n in N_omp]
        plt.plot(N_omp, T_omp, 's-', linewidth=2, label='OpenMP')
        
    if times_acc:
        N_acc = sorted(times_acc.keys())
        T_acc = [times_acc[n] for n in N_acc]
        plt.plot(N_acc, T_acc, '^-', linewidth=2, label='OpenACC')
    
    plt.xlabel('Grid Size N')
    plt.ylabel('Execution Time (s)')
    plt.title('Split-Step Fourier Method Performance Scaling')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.7)

    # plt.yscale('log')
    plt.xscale('log')

    all_n = set().union(
        times_serial.keys() if times_serial else set(),
        times_omp.keys() if times_omp else set(),
        times_acc.keys() if times_acc else set()
    )
    ticks = sorted(list(all_n))
    plt.xticks(ticks, [str(t) for t in ticks])
    plt.tight_layout()
    
    plt.savefig('benchmark_results.png', dpi=300)
    print("\nPlot saved to benchmark_results.png")

if __name__ == "__main__":
    main()
