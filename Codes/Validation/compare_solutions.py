import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_complex_array(filename):
    df = pd.read_csv(filename)
    return df['real'].values + 1j * df['imag'].values

def main():
    print("Loading validation data...")
    
    try:
        psi_serial = load_complex_array("serial_out.csv")
        psi_omp = load_complex_array("openmp_out.csv")
        psi_acc = load_complex_array("openacc_out.csv")
        # psi_fft = load_complex_array("fft_out.csv")
        # psi_cufft = load_complex_array("cufft_out.csv")
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # Calculate L-infinity norm (Max absolute difference)
    err_omp = np.max(np.abs(psi_serial - psi_omp))
    err_acc = np.max(np.abs(psi_serial - psi_acc))
    # err_fft = np.max(np.abs(psi_serial - psi_fft))
    # err_cufft = np.max(np.abs(psi_serial - psi_cufft))

    print("\n--- Validation Results (Max Absolute Error vs Serial) ---")
    print(f"OpenMP Naive  : {err_omp:.4e}")
    print(f"OpenACC Naive : {err_acc:.4e}")
    # print(f"FFTW3         : {err_fft:.4e}")
    # print(f"cuFFT         : {err_cufft:.4e}")
    print("---------------------------------------------------------")
    
    if max([err_omp, err_acc]) < 1e-10:
        print("PASS: All implementations produce identical solutions!\n")
    else:
        print("WARNING: Significant deviation detected.\n")

    # Plot
    N = len(psi_serial)
    L = 20.0
    t = np.linspace(-L/2, L/2, N, endpoint=False)

    plt.figure(figsize=(10, 6))
    
    # We plot dots for some, lines for others to see them overlap
    plt.plot(t, np.abs(psi_serial)**2, 'k-', linewidth=1, alpha = 0.8, label='Serial (Ground Truth)')
    plt.plot(t, np.abs(psi_omp)**2, 'r--', linewidth=2, alpha = 0.8, label='OpenMP')
    plt.plot(t, np.abs(psi_acc)**2, 'g:', linewidth=3, alpha = 0.8, label='OpenACC')
    
    # We plot every 10th point for FFT to avoid solid blocks of ink
    # skip = N // 50
    # plt.plot(t[::skip], np.abs(psi_fft[::skip])**2, 'bo', label='FFTW3')
    # plt.plot(t[::skip], np.abs(psi_cufft[::skip])**2, 'mx', markersize=8, label='cuFFT')
    
    plt.title("Cross-Validation: Final Pulse Intensity $|A(t)|^2$")
    plt.xlabel("Time (t)")
    plt.ylabel("Intensity")
    plt.xlim([-5, 5]) # Zoom in on the pulse
    plt.legend()
    plt.grid(True, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("solution_comparison.png", dpi=300)
    print("Plot saved to solution_comparison.png")

if __name__ == "__main__":
    main()
