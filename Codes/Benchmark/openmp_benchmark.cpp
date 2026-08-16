#include <iostream>
#include <vector>
#include <cmath>
#include <cstdio>
#include <chrono>
#include <omp.h>

const double PI = 3.14159265358979323846;

struct Complex {
    double real;
    double imag;
};

// 1D DFT routines

void forward_dft_1d(Complex* psi_in, Complex* psi_out, int N) {
    #pragma omp parallel for
    for (int k = 0; k < N; k++) {
        double sum_r = 0.0;
        double sum_i = 0.0;
        for (int t = 0; t < N; t++) {
            double angle = -2.0 * PI * (k * t) / (double)N;
            double weight_r = cos(angle);
            double weight_i = sin(angle);
            double in_r = psi_in[t].real;
            double in_i = psi_in[t].imag;
            sum_r += (in_r * weight_r) - (in_i * weight_i);
            sum_i += (in_r * weight_i) + (in_i * weight_r);
        }
        psi_out[k].real = sum_r;
        psi_out[k].imag = sum_i;
    }
}

void inverse_dft_1d(Complex* psi_in, Complex* psi_out, int N) {
    #pragma omp parallel for
    for (int t = 0; t < N; t++) {
        double sum_r = 0.0;
        double sum_i = 0.0;
        for (int k = 0; k < N; k++) {
            double angle = 2.0 * PI * (k * t) / (double)N;
            double weight_r = cos(angle);
            double weight_i = sin(angle);
            double in_r = psi_in[k].real;
            double in_i = psi_in[k].imag;
            sum_r += (in_r * weight_r) - (in_i * weight_i);
            sum_i += (in_r * weight_i) + (in_i * weight_r);
        }
        double norm = (double)N;
        psi_out[t].real = sum_r / norm;
        psi_out[t].imag = sum_i / norm;
    }
}

struct BenchmarkResult {
    double time;
    double prob_error;
};

// ASCII Art 1D Slice of the Wavefunction Probability

void print_ascii_slice(Complex* psi, int N, const char* title) {
    printf("\n  %s:\n", title);
    double max_prob = 0.0;
    for (int i = 0; i < N; i++) {
        double prob = psi[i].real * psi[i].real + psi[i].imag * psi[i].imag;
        if (prob > max_prob) max_prob = prob;
    }
    
    const int HEIGHT = 10;
    int skip = (N > 128) ? (N / 128) : 1; 

    for (int row = HEIGHT; row > 0; row--) {
        printf("  |");
        for (int i = 0; i < N; i += skip) {
            double prob = psi[i].real * psi[i].real + psi[i].imag * psi[i].imag;
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

// Run the SSFM benchmark for a given grid size N

BenchmarkResult run_benchmark(int N, int total_steps) {
    const double L = 20.0; 
    const double dt_sim = L / (double)N;
    const double dz = 0.02;
    const double gamma = 1.0;

    std::vector<Complex> h_psi(N);
    std::vector<Complex> h_psi_k(N);
    std::vector<double>  h_V(N);
    std::vector<double>  h_K(N);

    double initial_prob = 0.0;
    for (int i = 0; i < N; ++i) {
        double t = -L/2.0 + i * dt_sim;
        double sech_t = 1.0 / cosh(t);
        h_psi[i].real = sech_t;
        h_psi[i].imag = 0.0;

        int k_idx = (i < N/2) ? i : (i - N);
        double k = k_idx * (2.0 * PI / L);
        h_K[i] = 0.5 * k * k; 

        initial_prob += (h_psi[i].real * h_psi[i].real + h_psi[i].imag * h_psi[i].imag) * dt_sim;
    }

    Complex* psi   = h_psi.data();
    Complex* psi_k = h_psi_k.data();
    double*  V     = h_V.data();
    double*  K     = h_K.data();

    if (N <= 256) {
        print_ascii_slice(psi, N, "Initial Pulse");
    }

    auto t_start = std::chrono::high_resolution_clock::now();

    for (int s = 0; s < total_steps; ++s) {
        // Nonlinear half-step V = -gamma |psi|^2
        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            V[i] = -gamma * (psi[i].real * psi[i].real + psi[i].imag * psi[i].imag);
            double angle = -V[i] * dz / 2.0;
            double p_real = cos(angle);
            double p_imag = sin(angle);
            double psi_r = psi[i].real;
            double psi_i = psi[i].imag;
            psi[i].real = (psi_r * p_real) - (psi_i * p_imag);
            psi[i].imag = (psi_r * p_imag) + (psi_i * p_real);
        }

        forward_dft_1d(psi, psi_k, N);

        // Dispersion full-step
        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            double angle = -K[i] * dz;
            double p_real = cos(angle);
            double p_imag = sin(angle);
            double psi_r = psi_k[i].real;
            double psi_i = psi_k[i].imag;
            psi_k[i].real = (psi_r * p_real) - (psi_i * p_imag);
            psi_k[i].imag = (psi_r * p_imag) + (psi_i * p_real);
        }

        inverse_dft_1d(psi_k, psi, N);

        // Nonlinear half-step
        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            V[i] = -gamma * (psi[i].real * psi[i].real + psi[i].imag * psi[i].imag);
            double angle = -V[i] * dz / 2.0;
            double p_real = cos(angle);
            double p_imag = sin(angle);
            double psi_r = psi[i].real;
            double psi_i = psi[i].imag;
            psi[i].real = (psi_r * p_real) - (psi_i * p_imag);
            psi[i].imag = (psi_r * p_imag) + (psi_i * p_real);
        }
    }

    double total_prob = 0.0;
    #pragma omp parallel for reduction(+:total_prob)
    for (int i = 0; i < N; ++i) {
        total_prob += (psi[i].real * psi[i].real + psi[i].imag * psi[i].imag) * dt_sim;
    }
    
    auto t_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = t_end - t_start;
    
    if (N <= 256) {
        print_ascii_slice(psi, N, "Final Pulse");
    }
    
    return {elapsed.count(), std::abs(total_prob - initial_prob) / initial_prob};
}

// sweeping grid sizes

int main() {
    int grid_sizes[] = {512, 1024, 2048, 4096, 8192};
    int num_sizes = sizeof(grid_sizes) / sizeof(grid_sizes[0]);

    const int total_steps = 100;
    const int repeats = 1;

    printf("  Split-Step Fourier Method — OpenMP Benchmark (1D NLSE)\n");
    printf("  Time-steps per run : %d\n", total_steps);
    printf("  Repeats per size   : %d\n", repeats);

    printf("Running benchmarks\n");
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
            printf("    N=%3d  run %d/%d  =>  %.4f s  (Err: %.2e)\n", N, r+1, repeats, t, res.prob_error);
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
