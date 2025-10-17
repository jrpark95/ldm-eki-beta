// ldm_cram2.cu - CRAM48 Implementation
#include "ldm_cram2.cuh"
#include "../core/ldm.cuh"
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstdlib>

// ============================================================================
// CRAM48 Constants
// ============================================================================

const double ALPHA0_48 = 2.258038182743983e-47;

const double alpha_re_48[24] = {
    6.387380733878774e+2, 1.909896179065730e+2, 4.236195226571914e+2, 4.645770595258726e+2,
    7.765163276752433e+2, 1.907115136768522e+3, 2.909892685603256e+3, 1.944772206620450e+2,
    1.382799786972332e+5, 5.628442079602433e+3, 2.151681283794220e+2, 1.324720240514420e+3,
    1.617548476343347e+4, 1.112729040439685e+2, 1.074624783191125e+2, 8.835727765158191e+1,
    9.354078136054179e+1, 9.418142823531573e+1, 1.040012390717851e+2, 6.861882624343235e+1,
    8.766654491283722e+1, 1.056007619389650e+2, 7.738987569039419e+1, 1.041366366475571e+2
};

const double alpha_im_48[24] = {
    -6.743912502859256e+2, -3.973203432721332e+2, -2.041233768918671e+3, -1.652917287299683e+3,
    -1.783617639907328e+4, -5.887068595142284e+4, -9.953255345514560e+3, -1.427131226068449e+3,
    -3.256885197214938e+6, -2.924284515884309e+4, -1.121774011188224e+3, -6.370088443140973e+4,
    -1.008798413156542e+6, -8.837109731680418e+1, -1.457246116408180e+2, -6.388286188419360e+1,
    -2.195424319460237e+2, -6.719055740098035e+2, -1.693747595553868e+2, -1.177598523430493e+1,
    -4.596464999363902e+3, -1.738294585524067e+3, -4.311715386228984e+1, -2.777743732451969e+2
};

const double theta_re_48[24] = {
    -4.465731934165702e+1, -5.284616241568964e+0, -8.867715667624458e+0, 3.493013124279215e+0,
    1.564102508858634e+1, 1.742097597385893e+1, -2.834466755180654e+1, 1.661569367939544e+1,
    8.011836167974721e+0, -2.056267541998229e+0, 1.449208170441839e+1, 1.853807176907916e+1,
    9.932562704505182e+0, -2.244223871767187e+1, 8.590014121680897e-1, -1.286192925744479e+1,
    1.164596909542055e+1, 1.806076684783089e+1, 5.870672154659249e+0, -3.542938819659747e+1,
    1.901323489060250e+1, 1.885508331552577e+1, -1.734689708174982e+1, 1.316284237125190e+1
};

const double theta_im_48[24] = {
    6.233225190695437e+1, 4.057499381311059e+1, 4.325515754166724e+1, 3.281615453173585e+1,
    1.558061616372237e+1, 1.076629305714420e+1, 5.492841024648724e+1, 1.316994930024688e+1,
    2.780232111309410e+1, 3.794824788914354e+1, 1.799988210051809e+1, 5.974332563100539e+0,
    2.532823409972962e+1, 5.179633600312162e+1, 3.536456194294350e+1, 4.600304902833652e+1,
    2.287153304140217e+1, 8.368200580099821e+0, 3.029700159040121e+1, 5.834381701800013e+1,
    1.194282058271408e+0, 3.583428564427879e+0, 4.883941101108207e+1, 2.042951874827759e+1
};

// ============================================================================
// CSV Loading (Float Version)
// ============================================================================

bool load_A_matrix(const char* filename, float* A_matrix) {
    std::ifstream file(filename);
    if (!file.is_open()) return false;

    std::string line;
    for (int i = 0; i < N_NUCLIDES; ++i) {
        if (!std::getline(file, line)) {
            std::fill(A_matrix + i * N_NUCLIDES, A_matrix + N_NUCLIDES * N_NUCLIDES, 0.0f);
            break;
        }
        std::stringstream ss(line);
        std::string cell;
        for (int j = 0; j < N_NUCLIDES; ++j) {
            if (std::getline(ss, cell, ',')) {
                A_matrix[i * N_NUCLIDES + j] = cell.empty() ? 0.0f : std::stof(cell);
            } else {
                A_matrix[i * N_NUCLIDES + j] = 0.0f;
            }
        }
    }
    return true;
}

// ============================================================================
// CSV Loading (Double Version)
// ============================================================================

bool LDM::load_A_csv(const char* path, std::vector<double>& A_out) {
    A_out.assign(N_NUCLIDES * N_NUCLIDES, 0.0);
    std::ifstream f(path);
    if(!f.is_open()) return false;

    std::string line;
    int r = 0;
    while(std::getline(f, line) && r < N_NUCLIDES) {
        std::stringstream ss(line);
        std::string cell;
        int c = 0;
        while(std::getline(ss, cell, ',') && c < N_NUCLIDES) {
            A_out[r*N_NUCLIDES + c] = cell.empty() ? 0.0 : std::stod(cell);
            ++c;
        }
        ++r;
    }
    return true;
}

// ============================================================================
// Gaussian Elimination Solver
// ============================================================================

void LDM::gauss_solve_inplace(std::vector<double>& M, std::vector<double>& b, int n) {
    for(int k=0; k<n; k++){
        // Partial pivoting
        int piv = k;
        double pmax = std::fabs(M[k*n + k]);
        for(int i=k+1; i<n; i++){
            double v = std::fabs(M[i*n + k]);
            if(v > pmax){ pmax = v; piv = i; }
        }
        if(pmax < 1e-26) continue;

        // Swap rows if needed
        if(piv != k){
            for(int j=k; j<n; j++) std::swap(M[k*n + j], M[piv*n + j]);
            std::swap(b[k], b[piv]);
        }

        // Normalize pivot row
        double pivv = M[k*n + k];
        for(int j=k; j<n; j++) M[k*n + j] /= pivv;
        b[k] /= pivv;

        // Eliminate column
        for(int i=0; i<n; i++){
            if(i==k) continue;
            double f = M[i*n + k];
            if(std::fabs(f) < 1e-30) continue;
            for(int j=k; j<n; j++) M[i*n + j] -= f * M[k*n + j];
            b[i] -= f * b[k];
        }
    }
}

// ============================================================================
// CRAM48 Matrix Exponential Column Calculation
// ============================================================================

void LDM::cram48_expm_times_ej_host(const std::vector<double>& A,
                                    int j,
                                    std::vector<double>& col_out) {
    const int n = N_NUCLIDES;
    const int dim = 2*n;

    // dt from LDM class settings
    double local_dt = static_cast<double>(dt);

    // Unit vector e_j
    std::vector<double> n0(n, 0.0);
    n0[j] = 1.0;

    // B = dt * A
    std::vector<double> B(n*n);
    for(int i=0; i<n*n; i++) B[i] = local_dt * A[i];

    // Initialize result = e_j
    std::vector<double> result(n, 0.0);
    for(int i=0; i<n; i++) result[i] = n0[i];

    // Accumulate contributions from each pole
    for(int k=0; k<24; k++){
        double tr = theta_re_48[k], ti = theta_im_48[k];
        double ar = alpha_re_48[k], ai = alpha_im_48[k];

        // Build 2n×2n real block matrix M = (B - θ_k I)
        std::vector<double> M(dim*dim, 0.0);
        for(int r=0; r<n; r++){
            for(int c=0; c<n; c++){
                double Bij = B[r*n + c];
                M[r*dim + c]               = Bij;  // Top-left block
                M[(r+n)*dim + (c+n)]       = Bij;  // Bottom-right block
            }
            M[r*dim + r]           -= tr;  // Diagonal: -θ_re
            M[(r+n)*dim + (r+n)]   -= tr;
            M[r*dim + (r+n)]        =  ti; // Off-diagonal: ±θ_im
            M[(r+n)*dim + r]        = -ti;
        }

        // RHS = [result; 0]
        std::vector<double> b(dim, 0.0);
        for(int i=0; i<n; i++){ b[i] = result[i]; b[i+n] = 0.0; }

        // Solve (B - θ_k I) y = result
        gauss_solve_inplace(M, b, dim);

        // Accumulate: result += 2 * Re(α_k * y)
        for(int i=0; i<n; i++){
            double re = ar * b[i] - ai * b[i+n];
            result[i] += 2.0 * re;
        }
    }

    // Final scaling by α₀
    for(int i=0; i<n; i++){
        result[i] *= ALPHA0_48;
    }

    col_out = result;
}

// ============================================================================
// Build Transition Matrix and Upload to GPU
// ============================================================================

bool LDM::build_T_matrix_and_upload(const char* A60_csv_path) {
    std::vector<double> A;
    if(!load_A_csv(A60_csv_path, A)) return false;

    std::vector<float> Th(N_NUCLIDES * N_NUCLIDES, 0.0f);
    std::vector<double> col;

    // Compute each column of T = exp(dt*A)
    for(int j=0; j<N_NUCLIDES; j++){
        cram48_expm_times_ej_host(A, j, col);
        for(int r=0; r<N_NUCLIDES; r++){
            Th[r*N_NUCLIDES + j] = static_cast<float>(col[r]);
        }
    }

#ifdef DEBUG
    std::cout << Color::YELLOW << "[DEBUG] " << Color::RESET
              << "Allocating GPU memory for T matrix ("
              << Th.size() << " floats, " << Th.size()*sizeof(float) / 1024.0 << " KB)\n";
#endif

    // Free existing GPU memory if already allocated
    if (d_T_matrix != nullptr) {
        cudaFree(d_T_matrix);
        d_T_matrix = nullptr;
    }

    // Allocate GPU memory for T matrix
    size_t matrix_size = Th.size() * sizeof(float);
    cudaError_t e = cudaMalloc(&d_T_matrix, matrix_size);
    if (e != cudaSuccess) {
        std::cerr << Color::RED << Color::BOLD << "[ERROR] " << Color::RESET
                  << "Failed to allocate GPU memory for T matrix: " << cudaGetErrorString(e) << "\n";
        return false;
    }

    // Copy T matrix data to GPU
    e = cudaMemcpy(d_T_matrix, Th.data(), matrix_size, cudaMemcpyHostToDevice);
    if (e != cudaSuccess) {
        std::cerr << Color::RED << Color::BOLD << "[ERROR] " << Color::RESET
                  << "Failed to copy T matrix to GPU: " << cudaGetErrorString(e) << "\n";
        cudaFree(d_T_matrix);
        d_T_matrix = nullptr;
        return false;
    }

#ifdef DEBUG
    std::cout << Color::GREEN << "✓ " << Color::RESET
              << "T matrix copied to GPU memory\n";
#endif

    return true;
}

// ============================================================================
// Initialization Wrapper
// ============================================================================

bool LDM::initialize_cram_system(const char* A60_csv_path) {
    return build_T_matrix_and_upload(A60_csv_path);
}

// ============================================================================
// Device-Side Decay Calculation
// ============================================================================

// Note: cram_decay_calculation is now inline in ldm_cram2.cuh
