#include "Utility.h"
#include <stdexcept> 

using namespace std;

void printMatrix(const vector<vector<double>>& matrix){
    if(matrix.empty()){ cout << "[]\n"; return; }
    size_t n = matrix.size();
    size_t m = matrix[0].size();

    for(size_t i = 0; i<n; i++){
        for(size_t j = 0; j<m; j++){
            cout << matrix[i][j] << " ";
        }
        cout << "\n";
    }
}

vector<vector<double>> dot_product(
    const vector<vector<double>>& A,
    const vector<vector<double>>& B)
{
    if (A.empty() || A[0].empty() || B.empty() || B[0].empty()) {
        throw invalid_argument("Matrices cannot be empty.");
    }
    size_t L_A = A.size();
    size_t C_A = A[0].size();
    size_t L_B = B.size();
    size_t C_B = B[0].size();

    if (C_A != L_B) {
        throw invalid_argument("Incompatible dimensions for matrix multiplication (A.columns must equal B.rows).");
    }

    vector<vector<double>> C(L_A, vector<double>(C_B, 0.0));

    for (size_t i = 0; i < L_A; ++i) {
        for (size_t j = 0; j < C_B; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < C_A; ++k) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }

    return C;
}

// Add a bias vector (1 x out_features) to each row of a matrix (batch x out_features)
vector<vector<double>> add_bias_vector(
    const vector<vector<double>>& mat,
    const vector<double>& bias)
{
    if(mat.empty()) return {};
    size_t rows = mat.size();
    size_t cols = mat[0].size();
    if(bias.size() != cols) throw invalid_argument("Bias length must match number of columns in matrix");

    vector<vector<double>> out = mat;
    for(size_t i=0;i<rows;i++){
        for(size_t j=0;j<cols;j++){
            out[i][j] += bias[j];
        }
    }
    return out;
}