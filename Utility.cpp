// --- Utility.cpp (Corrected) ---
#include "Utility.h"
#include <cstddef>
#include <stdexcept> 
#include <vector>

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

// The dot_product function definition has been REMOVED from here!

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

vector<vector<double>> sub_matrices(
    const vector<vector<double>>& A,
    const vector<vector<double>>& B)
{
    size_t rows = A.size();
    size_t cols = A[0].size();
    if(rows != B.size() || cols != B[0].size()) throw invalid_argument("Incompatible dimensions of the matrices");

    vector<vector<double>> out = A;
    for(size_t i=0;i<rows;i++){
        for(size_t j=0;j<cols;j++){
            out[i][j] = A[i][j] - B[i][j];
        }
    }
    return out;
}


vector<vector<double>> constantMult(const vector<vector<double>>& mat, double k){
    size_t rows = mat.size();
    size_t cols = mat[0].size();

    vector<vector<double>> output = mat;

    for(int i = 0; i<rows; i++){
        for(int j = 0; j<cols; j++){
            output[i][j] = k*mat[i][j];
        }
    }

    return output;
}

vector<vector<double>> hadamard_product(
    const vector<vector<double>>& A,
    const vector<vector<double>>& B)
{
    int rows = A.size();
    int cols = A[0].size();
    vector<vector<double>> result(rows, vector<double>(cols, 0.0));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[i][j] = A[i][j] * B[i][j];
        }
    }

    return result;
}

double accuracy(vector<int> output, vector<int> expected){
    int m = output.size(), n = expected.size();
    double hits = 0.0;
    if (n != m) throw invalid_argument("Mismatch between output size and expected");

    for(int i = 0; i<n; i++){
        if(output[i] == expected[i]){
            hits++;
        }
    }

    return 100.0*(hits / m);
}