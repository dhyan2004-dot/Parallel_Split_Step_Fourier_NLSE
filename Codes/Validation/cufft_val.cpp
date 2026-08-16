#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
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

    cufftHandle plan_fwd, plan_inv;
    __acc_check_cufft(cufftPlan1d(&plan_fwd, N, CUFFT_Z2Z, 1), "plan_fwd");
    __acc_check_cufft(cufftPlan1d(&plan_inv, N, CUFFT_Z2Z, 1), "plan_inv");

    #pragma acc data copy(psi[0:N]) create(psi_k[0:N]) copyin(V[0:N], K[0:N])
    {
        for (int s = 0; s < total_steps; ++s) {
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

            #pragma acc host_data use_device(psi, psi_k)
            {
                cufftExecZ2Z(plan_fwd, (cufftDoubleComplex*)psi, (cufftDoubleComplex*)psi_k, CUFFT_FORWARD);
            }

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

            #pragma acc host_data use_device(psi_k, psi)
            {
                cufftExecZ2Z(plan_inv, (cufftDoubleComplex*)psi_k, (cufftDoubleComplex*)psi, CUFFT_INVERSE);
            }

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
    } 

    std::ofstream out("cufft_out.csv");
    out << "idx,real,imag\n";
    out << std::setprecision(15);
    for (int i = 0; i < N; ++i) {
        out << i << "," << psi[i].real << "," << psi[i].imag << "\n";
    }
    out.close();

    cufftDestroy(plan_fwd);
    cufftDestroy(plan_inv);

    std::cout << "Saved cufft_out.csv" << std::endl;
    return 0;
}
