#include <iostream>
#include <vector>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <fftw3.h>

const double PI = 3.14159265358979323846;

int main(int argc, char** argv) {
    std::string pulse_type = "sech";
    if (argc > 1) {
        pulse_type = argv[1];
    }

    int N = 4096;
    int total_steps = 1000;
    int save_every = 10;
    
    double L = 20.0;
    double dt_sim = L / (double)N;
    double dz = 0.01;
    
    double gamma = (pulse_type == "gaussian") ? 3.0 : 1.0;
    
    fftw_complex* psi   = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    fftw_complex* psi_k = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    
    std::vector<double> h_V(N);
    std::vector<double> h_K(N);

    // Initialising pulse
    for (int i = 0; i < N; ++i) {
        double t = -L/2.0 + i * dt_sim;
        if (pulse_type == "gaussian") {
            psi[i][0] = exp(-t * t / 2.0);
        } else {
            psi[i][0] = 1.0 / cosh(t);
        }
        psi[i][1] = 0.0;

        int k_idx = (i < N/2) ? i : (i - N);
        double k = k_idx * (2.0 * PI / L);
        h_K[i] = 0.5 * k * k; // Anomalous dispersion beta2 = -1
    }

    fftw_plan plan_fwd = fftw_plan_dft_1d(N, psi, psi_k, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan plan_inv = fftw_plan_dft_1d(N, psi_k, psi, FFTW_BACKWARD, FFTW_ESTIMATE);

    std::ofstream outfile("wavefunction_dynamics.csv");
    outfile << "z_step,t_idx,real,imag\n";

    for (int s = 0; s <= total_steps; ++s) {
        if (s % save_every == 0) {
            for (int i = 0; i < N; ++i) {
                outfile << s << "," << i << "," << psi[i][0] << "," << psi[i][1] << "\n";
            }
        }

        if (s == total_steps) break;

        // Nonlinear half-step
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

        // Inverse FFT scaling and Nonlinear half-step
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

    outfile.close();
    fftw_destroy_plan(plan_fwd);
    fftw_destroy_plan(plan_inv);
    fftw_free(psi);
    fftw_free(psi_k);

    printf("Dynamics exported to wavefunction_dynamics.csv.\n");
    return 0;
}
