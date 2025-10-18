#include "Utility.h"
#include "Linear.hpp"
#include <cassert>
#include <cmath>
#include <cstddef>
#include <vector>
#include <random>
#include <stdexcept>
//#include <iostream>

using namespace std;

template<typename T>
LinearLayer<T>::LinearLayer(int in_f, int out_f, Activation act): in_features(in_f), out_features(out_f), activation(act) {
    if(in_features <= 0 || out_features <= 0) throw invalid_argument("LinearLayer sizes must be positive");
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(-1.0, 1.0);

    weights.assign(in_features, vector<double>(out_features));

    for(int i=0;i<in_features;i++){
        for(int j=0;j<out_features;j++){
            weights[i][j] = dis(gen);
        }
    }

    biases.assign(out_features, 0.0);
    for(int i = 0; i<out_features; i++){
        biases[i] = dis(gen);
    }

    /* Only for debugging purposes
    cout << "Weights initialized:\n";
    printMatrix(weights);
    cout << "\nBiases initialized:\n";
    for (auto b : biases) cout << b << "\n";
    */
}

// forward: input is batch x in_features, returns batch x out_features
template<typename T>
vector<vector<double>> LinearLayer<T>::forward(const vector<vector<T>>& input){    if(input.empty()) return {}; 
    if(input[0].size() != static_cast<size_t>(in_features)) {
        throw invalid_argument("Input feature size does not match LinearLayer in_features");
    }
    vector<vector<double>> result = dot_product(input, weights); // (batch x out_features)
    result = add_bias_vector(result, biases);

    result = apply_activation(result, activation);

    return result;
}
