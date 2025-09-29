#include <cstddef>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

int num_classes = 10;

struct Dataset {
    vector<vector<uint8_t>> labels;  // one hot encoded labels
    vector<vector<uint32_t>> images;  // matrix of each pixel value in the image
    size_t dim = 0; // number of pixels of each image
    size_t n = 0; // number of images of the dataset
};


void to_one_hot(vector<uint8_t>&labels, int label){
    // This is correct
    if (label >= 0 && label < labels.size()) {
        labels[label] = 1;
    }
}

Dataset load_csv(const string& filename) {
    Dataset ReadMatrix;

    // Abre o arquivo para leitura (ifstream = input file stream)
    ifstream file(filename);

    // Verifica se o arquivo foi aberto com sucesso
    if (!file.is_open()) {
        cerr << "Erro ao abrir o arquivo: " << filename << endl;
        return ReadMatrix;
    }

    string line;
    size_t n = 0, dim;

    //ignores the header
    getline(file, line);

    while (getline(file, line)) {
        stringstream ss(line);
        string cell;
        
        vector<uint32_t> pixels;
        vector<uint8_t> labels(num_classes, 0);
        
        size_t current_cell_index = 0; // Use a dedicated index for the cell/column

        while (getline(ss, cell, ',')) {
            int value = stoi(cell);
            
            if(current_cell_index == 0){
                // First column is the label
                to_one_hot(labels, value);
                ReadMatrix.labels.push_back(labels);

            } else {
                // Subsequent columns are pixels
                pixels.push_back(value);
            }

            current_cell_index++;   
        }

        // Only store 'dim' for the first successful read to get the column count
        if (n == 0) {
             dim = current_cell_index;
        }

        ReadMatrix.images.push_back(pixels);
        n++;
    }
    
    ReadMatrix.n = n;
    ReadMatrix.dim = dim - 1; 

    file.close();

    return ReadMatrix;
}


int main(int argc, char* argv[]) {

    if(argc < 3){
        cerr << "Please provide the train.csv and test.csv" << endl;
        return 1;
    }
    
    cout << "[INFO] Reading files..." << endl;
    Dataset train =  load_csv(argv[1]);
    Dataset test =  load_csv(argv[2]);

    cout << "Reading completed:" << endl;
    cout << "Length of train.csv: " << train.n << endl;
    cout << "Length of test.csv: " << test.n << endl;

    return 0;
}