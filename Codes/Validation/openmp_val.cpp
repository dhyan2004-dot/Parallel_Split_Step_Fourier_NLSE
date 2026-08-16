#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <omp.h>

const double PI = 3.14159265358979323846;

struct Complex {
    double real;
    double imag;
};

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

int main() {
    int N = 4096;
    int total_steps = 100;
    double L = 20.0; 
    double dt_sim = L / (double)N;
    double dz = 0.02;
    double gamma = 1.0;

    std::vector<Complex> h_psi(N);
    std::vector<Complex> h_psi_k(N);
    std::vector<double>  h_V(N);
    std::vector<double>  h_K(N);

    for (int i = 0; i < N; ++i) {
        double t = -L/2.0 + i * dt_sim;
        double sech_t = 1.0 / cosh(t);
        h_psi[i].real = sech_t;
        h_psi[i].imag = 0.0;

        int k_idx = (i < N/2) ? i : (i - N);
        double k = k_idx * (2.0 * PI / L);
        h_K[i] = 0.5 * k * k; 
    }

    Complex* psi   = h_psi.data();
    Complex* psi_k = h_psi_k.data();
    double*  V     = h_V.data();
    double*  K     = h_K.data();

    for (int s = 0; s < total_steps; ++s) {
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

    std::ofstream out("openmp_out.csv");
    out << "idx,real,imag\n";
    out << std::setprecision(15);
    for (int i = 0; i < N; ++i) {
        out << i << "," << psi[i].real << "," << psi[i].imag << "\n";
    }
    out.close();

    std::cout << "Saved openmp_out.csv" << std::endl;
    return 0;
}
