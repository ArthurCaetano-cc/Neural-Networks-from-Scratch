#include "Activations.h"
#include "Utility.h"
#include <cassert>
#include <cmath>
#include <cstddef>
#include <vector>
#include <stdexcept>

using namespace std;

class LinearLayer {
private:
    int in_features;
    int out_features;
    Activation activation;

public:
    vector<vector<double>> weights; // shape: in_features x out_features
    vector<double> biases; // length: out_features
    vector<vector<double>> z; //output of the forward propagation, stored to calculate the gradients

    LinearLayer(int in_f, int out_f, Activation act = Activation::None);

    // forward: input is batch x in_features, returns batch x out_features
    template<typename T>
    vector<vector<double>> forward(const vector<vector<T>>& input){
        if(input[0].size() != static_cast<size_t>(in_features)) {
            throw invalid_argument("Input feature size does not match LinearLayer in_features");
        }
        vector<vector<double>> result = dot_product(input, weights); // (batch x out_features)
        result = add_bias_vector(result, biases);

        result = apply_activation(result, activation);
        z = result;

        return result;
    }
};
