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
    print("Compiling ALL FFT benchmarks")
    run_cmd("g++ -O3 fft_serial_benchmark.cpp -o fft_serial_benchmark -lfftw3")
    run_cmd("nvc++ -acc -fast -cudalib=cufft cufft_benchmark.cpp -o cufft_benchmark")
    
    times_fft = run_and_parse("fft_serial_benchmark")
    times_cufft = run_and_parse("cufft_benchmark")
    
    plt.figure(figsize=(10, 6))
        
    if times_fft:
        N_fft = sorted(times_fft.keys())
        T_fft = [times_fft[n] for n in N_fft]
        plt.plot(N_fft, T_fft, 'x-', linewidth=2, label='FFTW3 CPU')
        
    if times_cufft:
        N_cufft = sorted(times_cufft.keys())
        T_cufft = [times_cufft[n] for n in N_cufft]
        plt.plot(N_cufft, T_cufft, 'o-', linewidth=2, label='cuFFT GPU')
    
    plt.xlabel('Grid Size N')
    plt.ylabel('Average Execution Time (s)')
    plt.title('Split-Step Fourier Method - FFT Performance Scaling')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.7)
    
    # plt.yscale('log')
    plt.xscale('log')
    
    import matplotlib.ticker as ticker
    plt.gca().xaxis.set_minor_formatter(ticker.NullFormatter())

    all_n = set().union(
        times_fft.keys() if times_fft else set(),
        times_cufft.keys() if times_cufft else set()
    )
    ticks = sorted(list(all_n))
    plt.xticks(ticks, [str(t) for t in ticks])

    plt.tight_layout()
    
    plt.savefig('fft_benchmark_results.png', dpi=300)
    print("\nPlot saved to fft_benchmark_results.png")

if __name__ == "__main__":
    main()
