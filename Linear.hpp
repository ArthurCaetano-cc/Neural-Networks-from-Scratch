#include "Activations.h"
#include <cassert>
#include <cmath>
#include <vector>

using namespace std;

template<typename T>
class LinearLayer {
private:
    int in_features;
    int out_features;
    Activation activation;

public:
    vector<vector<double>> weights; // shape: in_features x out_features
    vector<double> biases; // length: out_features

    LinearLayer(int in_f, int out_f, Activation act = Activation::None);

    // forward: input is batch x in_features, returns batch x out_features
    vector<vector<double>> forward(const vector<vector<T>>& input);
};
