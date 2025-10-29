#pragma once
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

struct Dataset {
    vector<vector<double>> labels;  // one hot encoded labels
    vector<vector<uint32_t>> images;  // matrix of each pixel value in the image
    size_t dim = 0; // number of pixels of each image
    size_t n = 0; // number of images of the dataset
};

void to_one_hot(vector<double>&labels, int label);

Dataset load_csv(const string& filename);