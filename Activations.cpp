#include "Activations.h"

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

/* SOFTMAX */

double exp_sum(vector<double> z){
    double sum = 0.0;
    for(int i = 0; i<z.size(); i++){
        sum += exp(z[i]);
    }
    
    return sum;
}

vector<vector<double>> softmax(const vector<vector<double>>z){

    int rows = z.size(), columns = z[0].size();
    vector<vector<double>> output(z.size());
    
    for(int i = 0; i<rows; i++){
        output[i].resize(columns);
    }

    for(int i = 0; i<rows; i++){
        double row_sum = exp_sum(z[i]);

        for(int j = 0; j<columns; j++){
            output[i][j] = (exp(z[i][j]) / row_sum);
        }

    }

    return output;
}

vector<vector<double>> apply_activation(vector<vector<double>> z, Activation activation){
    switch (activation) {
        case Activation::ReLU: return relu(z);
        case Activation::Softmax: return softmax(z);
        default: return z;
    }
}

/* Sigmoid (not implemented yet)*/

/* Tanh (not implemented yet) */
