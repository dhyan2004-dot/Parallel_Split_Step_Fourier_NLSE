#include <iostream>
#include <vector>
#include <cmath>
#include <cstdio>
#include <chrono>
#include <cufft.h>
#include <openacc.h>

const double PI = 3.14159265358979323846;

struct Complex {
    double real;
    double imag;
};

void __acc_check_cufft(cufftResult res, const char* msg) {
    if (res != CUFFT_SUCCESS) {
        std::cerr << "CUFFT error at " << msg << ": " << res << std::endl;
        exit(1);
    }
}

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

// GPU warm-up

void gpu_warmup() {
    const int WARM_N = 256;
    std::vector<Complex> warm_psi(WARM_N);
    std::vector<Complex> warm_psi_k(WARM_N);

    for (int i = 0; i < WARM_N; i++) {
        warm_psi[i].real = 1.0;
        warm_psi[i].imag = 0.0;
    }

    Complex* psi   = warm_psi.data();
    Complex* psi_k = warm_psi_k.data();

    cufftHandle plan_fwd, plan_inv;
    if (cufftPlan1d(&plan_fwd, WARM_N, CUFFT_Z2Z, 1) != CUFFT_SUCCESS) {
        std::cerr << "[Warmup] Failed to create forward plan." << std::endl;
        return;
    }
    if (cufftPlan1d(&plan_inv, WARM_N, CUFFT_Z2Z, 1) != CUFFT_SUCCESS) {
        std::cerr << "[Warmup] Failed to create inverse plan." << std::endl;
        return;
    }

    #pragma acc data copy(psi[0:WARM_N]) create(psi_k[0:WARM_N])
    {
        #pragma acc host_data use_device(psi, psi_k)
        {
            cufftExecZ2Z(plan_fwd, (cufftDoubleComplex*)psi, (cufftDoubleComplex*)psi_k, CUFFT_FORWARD);
            cufftExecZ2Z(plan_inv, (cufftDoubleComplex*)psi_k, (cufftDoubleComplex*)psi, CUFFT_INVERSE);
        }
    }

    cufftDestroy(plan_fwd);
    cufftDestroy(plan_inv);

    printf("cuFFT GPU warm-up complete\n\n");
}

struct BenchmarkResult {
    double time;
    double prob_error;
};

// Running the SSFM benchmark for a given grid size N

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

    cufftHandle plan_fwd, plan_inv;
    __acc_check_cufft(cufftPlan1d(&plan_fwd, N, CUFFT_Z2Z, 1), "plan_fwd");
    __acc_check_cufft(cufftPlan1d(&plan_inv, N, CUFFT_Z2Z, 1), "plan_inv");

    if (N <= 256) {
        print_ascii_slice(psi, N, "Initial Pulse");
    }

    auto t_start = std::chrono::high_resolution_clock::now();

    double last_total_prob = 0.0;
    
    #pragma acc data copy(psi[0:N]) create(psi_k[0:N]) copyin(V[0:N], K[0:N])
    {
        for (int s = 0; s < total_steps; ++s) {

            // Nonlinear half-step V = -gamma |psi|^2
            #pragma acc parallel loop
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

            // Forward DFT via CUFFT
            #pragma acc host_data use_device(psi, psi_k)
            {
                cufftExecZ2Z(plan_fwd, (cufftDoubleComplex*)psi, (cufftDoubleComplex*)psi_k, CUFFT_FORWARD);
            }

            // Dispersion full-step
            #pragma acc parallel loop
            for (int i = 0; i < N; ++i) {
                double angle = -K[i] * dz;
                double p_real = cos(angle);
                double p_imag = sin(angle);
                double psi_r = psi_k[i].real;
                double psi_i = psi_k[i].imag;
                psi_k[i].real = (psi_r * p_real) - (psi_i * p_imag);
                psi_k[i].imag = (psi_r * p_imag) + (psi_i * p_real);
            }

            // Inverse DFT via CUFFT
            #pragma acc host_data use_device(psi_k, psi)
            {
                cufftExecZ2Z(plan_inv, (cufftDoubleComplex*)psi_k, (cufftDoubleComplex*)psi, CUFFT_INVERSE);
            }

            // CUFFT inverse normalization and Nonlinear half-step
            double norm = (double)N;
            #pragma acc parallel loop
            for (int i = 0; i < N; ++i) {
                psi[i].real /= norm;
                psi[i].imag /= norm;
                
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
        #pragma acc parallel loop reduction(+:total_prob)
        for (int i = 0; i < N; ++i) {
            total_prob += (psi[i].real * psi[i].real + psi[i].imag * psi[i].imag) * dt_sim;
        }
        
        last_total_prob = total_prob;
    } // End of #pragma acc data

    auto t_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = t_end - t_start;
    
    if (N <= 256) {
        print_ascii_slice(psi, N, "Final Pulse");
    }
    
    cufftDestroy(plan_fwd);
    cufftDestroy(plan_inv);

    return {elapsed.count(), std::abs(last_total_prob - initial_prob) / initial_prob};
}

int main() {
    int grid_sizes[] = {512, 1024, 2048, 4096, 8192, 16384, 32768, 65536}; 
    int num_sizes = sizeof(grid_sizes) / sizeof(grid_sizes[0]);

    const int total_steps = 100;
    const int repeats = 3;
    printf("  Split-Step Fourier Method — cuFFT Benchmark (1D NLSE)\n");
    printf("  Time-steps per run : %d\n", total_steps);
    printf("  Repeats per size   : %d\n", repeats);

    printf("Warming up cuFFT and GPU device\n");
    gpu_warmup();

    printf("Running benchmarks\n");
    printf("%-10s  %-12s  %-14s  %-14s  %-14s  %-15s\n",
           "N", "Grid (N)", "Avg Time (s)", "Min Time (s)", "Max Time (s)", "Prob Error");

    for (int g = 0; g < num_sizes; g++) {
        int N = grid_sizes[g];
        double t_min = 1e30;
        double t_max = 0.0;
        double t_sum = 0.0;
        double last_prob_error = 0.0;

        printf("Benchmarking N = %d \n", N);
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
