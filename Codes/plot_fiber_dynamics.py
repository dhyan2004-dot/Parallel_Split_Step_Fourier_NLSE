import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14, 'axes.labelsize': 14, 'xtick.labelsize': 12, 'ytick.labelsize': 12, 'axes.titlesize': 16, 'legend.fontsize': 12})

def main():
    print("Loading data from wavefunction_dynamics.csv")
    df = pd.read_csv("wavefunction_dynamics.csv")
    
    # Reconstructing the 2D array
    z_steps = df['z_step'].unique()
    t_idx = df['t_idx'].unique()
    
    N = len(t_idx)
    M = len(z_steps)
    
    L = 40.0
    t = np.linspace(-L/2, L/2, N, endpoint=False)
    
    # Pivoting to create matrices
    real_part = df.pivot(index='z_step', columns='t_idx', values='real').values
    imag_part = df.pivot(index='z_step', columns='t_idx', values='imag').values
    
    psi_matrix = real_part + 1j * imag_part
    intensity = np.abs(psi_matrix)**2
    
    # Plot 1: Waterfall plot
    fig_wf, ax_wf = plt.subplots(figsize=(8, 6))
    extent = [t[0], t[-1], z_steps[-1], z_steps[0]]
    im = ax_wf.imshow(intensity, aspect='auto', extent=extent, cmap='magma', origin='upper')
    ax_wf.set_title("Waterfall Plot: Intensity $|A(t,z)|^2$")
    ax_wf.set_xlabel("Time (t)")
    ax_wf.set_ylabel("Propagation Step (z)")
    ax_wf.set_xlim([-10, 10])
    fig_wf.colorbar(im, ax=ax_wf, label="Intensity")
    plt.tight_layout()
    fig_wf.savefig("plot_waterfall.png", dpi=300)
    plt.close(fig_wf)
    print("Saved plot_waterfall.png")
    
    # Plot 2: Pulse Dynamics at Different Z
    num_snapshots = 4
    snapshot_indices = np.linspace(0, M - 1, num_snapshots, dtype=int)
    
    fig_dyn, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    
    dz = 0.01
    
    for ax, idx in zip(axes, snapshot_indices):
        z_step = z_steps[idx]
        z_real = z_step * dz
        
        ax.plot(t, intensity[idx, :], color='black', linewidth=1.2)
        ax.set_title(f"z = {z_real:.1f}")
        ax.set_xlim([-10, 10])
        ax.tick_params(direction='in')
        
    for ax in axes[2:]:
        ax.set_xlabel("Time (T)")
    for ax in [axes[0], axes[2]]:
        ax.set_ylabel("Intensity")
    for ax in axes:
        ax.set_ylim([0, 1])
        ax.grid(True, linestyle=':', alpha=0.5)
    fig_dyn.suptitle("Pulse Propagation Dynamics", fontsize=18)
    plt.tight_layout()
    fig_dyn.savefig("plot_pulse_dynamics.png", dpi=300)
    plt.close(fig_dyn)
    print("Saved plot_pulse_dynamics.png")

    # Plot 3: Spectral Broadening
    fig_spec, ax_spec = plt.subplots(figsize=(8, 6))
    dt = L / N
    freq = np.fft.fftfreq(N, d=dt)
    freq = np.fft.fftshift(freq)
    
    psi_start = psi_matrix[0, :]
    psi_end = psi_matrix[-1, :]
    
    spec_start = np.abs(np.fft.fftshift(np.fft.fft(psi_start)))**2
    spec_end = np.abs(np.fft.fftshift(np.fft.fft(psi_end)))**2
    
    ax_spec.plot(freq, spec_start / np.max(spec_start), label='Start (z=0)', linestyle='--', color='gray', linewidth=2)
    ax_spec.plot(freq, spec_end / np.max(spec_end), label='End (z=L)', color='purple', linewidth=1, alpha = 0.5)
    ax_spec.set_title("Spectral Broadening (SPM)")
    ax_spec.set_xlabel("Frequency")
    ax_spec.set_ylabel("Normalized Power Spectrum")
    ax_spec.set_xlim([-1, 1]) 
    ax_spec.set_yscale('log')
    ax_spec.set_ylim([1e-4, 1.2])
    ax_spec.legend()
    ax_spec.grid(True, alpha=0.5)
    plt.tight_layout()
    fig_spec.savefig("plot_spectral_broadening.png", dpi=300)
    plt.close(fig_spec)
    print("Saved plot_spectral_broadening.png")
    
    # Plot 4: Pulse Chirp
    fig_chirp, ax_chirp = plt.subplots(figsize=(8, 6))
    
    # Chirp is defined as -d(phi)/dt
    phase = np.unwrap(np.angle(psi_end))
    chirp = -np.gradient(phase, dt)
    
    ax_chirp_twin = ax_chirp.twinx()
    
    ax_chirp.plot(t, np.abs(psi_end)**2, 'b-', label='Intensity', linewidth=2)
    ax_chirp.set_ylabel("Intensity", color='b')
    ax_chirp.tick_params(axis='y', labelcolor='b')
    
    # Only plotting chirp where intensity is significant to avoid noise from trailing tails
    threshold = 0.05 * np.max(np.abs(psi_end)**2)
    valid_idx = np.abs(psi_end)**2 > threshold
    
    # Filtering the chirp
    chirp_clean = np.where(valid_idx, chirp, np.nan)
    
    ax_chirp_twin.plot(t, chirp_clean, 'r--', label='Chirp', linewidth=1, alpha = 0.4)
    ax_chirp_twin.set_ylabel("Chirp (Inst. Frequency)", color='r')
    ax_chirp_twin.tick_params(axis='y', labelcolor='r')
    
    ax_chirp.set_title("Final Pulse: Intensity & Chirp")
    ax_chirp.set_xlabel("Time (t)")
    ax_chirp.set_xlim([-10, 10])
    
    plt.tight_layout()
    fig_chirp.savefig("plot_pulse_chirp.png", dpi=300)
    plt.close(fig_chirp)
    print("Saved plot_pulse_chirp.png")

if __name__ == "__main__":
    main()
