// --- Utility.h (Corrected) ---
#pragma once
#include <iostream>
#include <vector>
#include <cstdint>
#include <stdexcept> // Include for invalid_argument in template

using namespace std;

void printMatrix(const vector<vector<double>>& matrix);

template<typename T>
vector<vector<double>> dot_product(
    const vector<vector<T>>& A,
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
                // T * double implicitly casts T to double, which is fine for uint32_t
                sum += (double)A[i][k] * B[k][j]; 
            }
            C[i][j] = sum;
        }
    }

    return C;
}

template<typename T>
vector<vector<double>> transpoose(const vector<vector<T>>& mat){
    size_t rows = mat.size();
    size_t cols = mat[0].size();
    vector<vector<double>> matT(cols, vector<double>(rows));;

    for(int i = 0; i<rows; i++){
        for(int j = 0; j<cols; j++){
            matT[j][i] = mat[i][j];
        }
    }

    return matT;
}

vector<vector<double>> add_bias_vector(
    const vector<vector<double>>& mat,
    const vector<double>& bias);

vector<vector<double>> sub_matrices(const vector<vector<double>>& A, const vector<vector<double>>& B);

vector<vector<double>> constantMult(const vector<vector<double>>& mat, double k);

vector<vector<double>> hadamard_product(
    const vector<vector<double>>& A,
    const vector<vector<double>>& B);

double accuracy(vector<int> output, vector<int> expected);