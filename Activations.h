#pragma once
#include <vector>
#include <cmath>

using namespace std;

enum class Activation {
    None,
    ReLU,
    Sigmoid,
    Tanh,
    Softmax
};
 
/* RELU */

vector<vector<double>> relu(const vector<vector<double>>z);
vector<vector<double>> ReLU_derivative(const vector<vector<double>>& Z);

/* SOFTMAX */

double exp_sum(vector<double> z);

vector<vector<double>> softmax_fn(const vector<vector<double>>& z);

vector<vector<double>> apply_activation(vector<vector<double>> z, Activation activation);

/* Sigmoid (not implemented yet)*/

/* Tanh (not implemented yet) */