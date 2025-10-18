#pragma once
#include <iostream>
#include <vector>

using namespace std;

void printMatrix(const vector<vector<double>>& matrix);

vector<vector<double>> dot_product(
    const vector<vector<double>>& A,
    const vector<vector<double>>& B);

// Add a bias vector (1 x out_features) to each row of a matrix (batch x out_features)
vector<vector<double>> add_bias_vector(
    const vector<vector<double>>& mat,
    const vector<double>& bias);