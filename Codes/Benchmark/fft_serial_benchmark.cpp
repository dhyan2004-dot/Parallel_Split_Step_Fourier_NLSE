#include <iostream>
#include <vector>
#include <cmath>
#include <cstdio>
#include <chrono>
#include <fftw3.h>

const double PI = 3.14159265358979323846;

struct BenchmarkResult {
    double time;
    double prob_error;
};

// ASCII Art 1D Slice of the Wavefunction Probability

void print_ascii_slice(fftw_complex* psi, int N, const char* title) {
    printf("\n  %s:\n", title);
    double max_prob = 0.0;
    for (int i = 0; i < N; i++) {
        double prob = psi[i][0] * psi[i][0] + psi[i][1] * psi[i][1];
        if (prob > max_prob) max_prob = prob;
    }
    
    const int HEIGHT = 10;
    int skip = (N > 128) ? (N / 128) : 1; 

    for (int row = HEIGHT; row > 0; row--) {
        printf("  |");
        for (int i = 0; i < N; i += skip) {
            double prob = psi[i][0] * psi[i][0] + psi[i][1] * psi[i][1];
            int h = (int)((prob / max_prob) * HEIGHT);
            if (h >= row) {
                printf("*");
            } else {
                printf(" ");
            }
        }
        printf("\n");
    }
    printf("  +");
    for (int i = 0; i < N; i += skip) printf("-");
    printf("\n\n");
}

// Running the SSFM benchmark for a given grid size N

BenchmarkResult run_benchmark(int N, int total_steps) {
    const double L = 20.0; 
    const double dt_sim = L / (double)N;
    const double dz = 0.02;
    const double gamma = 1.0;

    fftw_complex* psi   = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    fftw_complex* psi_k = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    
    std::vector<double>  h_V(N);
    std::vector<double>  h_K(N);

    double initial_prob = 0.0;
    for (int i = 0; i < N; ++i) {
        double t = -L/2.0 + i * dt_sim;
        double sech_t = 1.0 / cosh(t);
        psi[i][0] = sech_t;
        psi[i][1] = 0.0;

        int k_idx = (i < N/2) ? i : (i - N);
        double k = k_idx * (2.0 * PI / L);
        h_K[i] = 0.5 * k * k; 

        initial_prob += (psi[i][0] * psi[i][0] + psi[i][1] * psi[i][1]) * dt_sim;
    }

    fftw_plan plan_fwd = fftw_plan_dft_1d(N, psi, psi_k, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan plan_inv = fftw_plan_dft_1d(N, psi_k, psi, FFTW_BACKWARD, FFTW_ESTIMATE);

    if (N <= 256) {
        print_ascii_slice(psi, N, "Initial Pulse");
    }

    auto t_start = std::chrono::high_resolution_clock::now();

    for (int s = 0; s < total_steps; ++s) {
        // Nonlinear half-step V = -gamma |psi|^2
        for (int i = 0; i < N; ++i) {
            h_V[i] = -gamma * (psi[i][0] * psi[i][0] + psi[i][1] * psi[i][1]);
            double angle = -h_V[i] * dz / 2.0;
            double p_real = cos(angle);
            double p_imag = sin(angle);
            double psi_r = psi[i][0];
            double psi_i = psi[i][1];
            psi[i][0] = (psi_r * p_real) - (psi_i * p_imag);
            psi[i][1] = (psi_r * p_imag) + (psi_i * p_real);
        }

        fftw_execute(plan_fwd);

        // Dispersion full-step
        for (int i = 0; i < N; ++i) {
            double angle = -h_K[i] * dz;
            double p_real = cos(angle);
            double p_imag = sin(angle);
            double psi_r = psi_k[i][0];
            double psi_i = psi_k[i][1];
            psi_k[i][0] = (psi_r * p_real) - (psi_i * p_imag);
            psi_k[i][1] = (psi_r * p_imag) + (psi_i * p_real);
        }

        fftw_execute(plan_inv);

        // Normalize inverse FFT and Nonlinear half-step
        double norm = (double)N;
        for (int i = 0; i < N; ++i) {
            psi[i][0] /= norm;
            psi[i][1] /= norm;
            
            h_V[i] = -gamma * (psi[i][0] * psi[i][0] + psi[i][1] * psi[i][1]);
            double angle = -h_V[i] * dz / 2.0;
            double p_real = cos(angle);
            double p_imag = sin(angle);
            double psi_r = psi[i][0];
            double psi_i = psi[i][1];
            psi[i][0] = (psi_r * p_real) - (psi_i * p_imag);
            psi[i][1] = (psi_r * p_imag) + (psi_i * p_real);
        }
    }

    double total_prob = 0.0;
    for (int i = 0; i < N; ++i) {
        total_prob += (psi[i][0] * psi[i][0] + psi[i][1] * psi[i][1]) * dt_sim;
    }
    
    auto t_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = t_end - t_start;
    
    if (N <= 256) {
        print_ascii_slice(psi, N, "Final Pulse");
    }
    
    fftw_destroy_plan(plan_fwd);
    fftw_destroy_plan(plan_inv);
    fftw_free(psi);
    fftw_free(psi_k);
    
    return {elapsed.count(), std::abs(total_prob - initial_prob) / initial_prob};
}

//sweeping grid sizes

int main() {
    // For FFT, we can test much larger grids
    int grid_sizes[] = {512, 1024, 2048, 4096, 8192, 16384, 32768, 65536}; 
    int num_sizes = sizeof(grid_sizes) / sizeof(grid_sizes[0]);

    const int total_steps = 100;
    const int repeats = 3;
    printf("  Split-Step Fourier Method — FFT Serial Benchmark (1D NLSE)\n");
    printf("  Time-steps per run : %d\n", total_steps);
    printf("  Repeats per size   : %d\n", repeats);
    printf("Running benchmarks...\n\n");
    printf("%-10s  %-12s  %-14s  %-14s  %-14s  %-15s\n",
           "N", "Grid (N)", "Avg Time (s)", "Min Time (s)", "Max Time (s)", "Prob Error");

    for (int g = 0; g < num_sizes; g++) {
        int N = grid_sizes[g];
        double t_min = 1e30;
        double t_max = 0.0;
        double t_sum = 0.0;
        double last_prob_error = 0.0;

        printf("Benchmarking N = %d\n", N);
        for (int r = 0; r < repeats; r++) {
            BenchmarkResult res = run_benchmark(N, total_steps);
            double t = res.time;
            last_prob_error = res.prob_error;
            
            t_sum += t;
            if (t < t_min) t_min = t;
            if (t > t_max) t_max = t;
            printf("    N=%5d  run %d/%d  =>  %.4f s  (Err: %.2e)\n", N, r+1, repeats, t, res.prob_error);
        }

        double t_avg = t_sum / repeats;
        char grid_label[32];
        snprintf(grid_label, sizeof(grid_label), "%d", N);
        printf("%-10d  %-12s  %-14.4f  %-14.4f  %-14.4f  %-15.2e\n\n",
               N, grid_label, t_avg, t_min, t_max, last_prob_error);
    }

    printf("  Benchmark complete.\n");

    return 0;
}
