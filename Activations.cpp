#include "Activations.h"
#include <algorithm>

using namespace std;
 
/* RELU */

vector<vector<double>> relu(const vector<vector<double>>z){
    int rows = z.size(), columns = z[0].size();
    vector<vector<double>> output(z.size());

    for(int i = 0; i<rows; i++){
        output[i].resize(columns);
    }

    for(int i = 0; i<rows; i++){
        for(int j = 0; j<columns; j++){
            output[i][j] = z[i][j] >= 0 ? z[i][j] : 0;
        }
    }
    return output;
}

vector<vector<double>> ReLU_derivative(const vector<vector<double>>& Z)
{
    int rows = Z.size();
    int cols = Z[0].size();
    vector<vector<double>> result(rows, vector<double>(cols, 0.0));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[i][j] = Z[i][j] > 0.0 ? 1.0 : 0.0;
        }
    }

    return result;
}

/* SOFTMAX */

double exp_sum(vector<double> z){
    double sum = 0.0;
    for(int i = 0; i < z.size(); i++){
        sum += exp(z[i]); // FIXED: Using 'z[i]'
    }
    return sum;
}

vector<vector<double>> softmax_fn(const vector<vector<double>>& z) {
    if (z.empty() || z[0].empty()) return {};

    int rows = z.size();
    int columns = z[0].size();
    vector<vector<double>> output(rows, vector<double>(columns));

    for (int i = 0; i < rows; i++) {
        double max_val = *max_element(z[i].begin(), z[i].end());
        double sum_exp = 0.0;

        for (int j = 0; j < columns; j++)
            sum_exp += exp(z[i][j] - max_val);

        for (int j = 0; j < columns; j++)
            output[i][j] = exp(z[i][j] - max_val) / sum_exp;
    }

    return output;
}

vector<vector<double>> apply_activation(vector<vector<double>> z, Activation activation){
    switch (activation) {
        case Activation::ReLU: return relu(z);
        case Activation::Softmax: return softmax_fn(z);
        default: return z;
    }
}

/* Sigmoid (not implemented yet)*/

/* Tanh (not implemented yet) */
