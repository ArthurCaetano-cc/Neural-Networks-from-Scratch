#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <random> 

using namespace std;

// CSV Reading

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

vector<vector<double>> dot_product(
    const vector<vector<double>>& A, 
    const vector<vector<double>>& B) 
{
    // 1. Obter dimensões das matrizes
    if (A.empty() || A[0].empty() || B.empty() || B[0].empty()) {
        // Trata o caso de matrizes vazias ou com dimensão zero
        throw invalid_argument("As matrizes nao podem ser vazias.");
    }
    size_t L_A = A.size();
    size_t C_A = A[0].size();

    size_t L_B = B.size();
    size_t C_B = B[0].size();
    assert(C_A == L_B && "As colunas de A devem ser iguais as linhas de B para multiplicacao.");

    if (C_A != L_B) {
        throw invalid_argument(
            "Dimensoes incompativeis: Colunas de A (" + to_string(C_A) + 
            ") devem ser iguais as Linhas de B (" + to_string(L_B) + ").");
    }

    vector<vector<double>> C(L_A, vector<double>(C_B, 0.0));


    for (size_t i = 0; i < L_A; ++i) { 
        for (size_t j = 0; j < C_B; ++j) { 
            for (size_t k = 0; k < C_A; ++k) { 
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return C;
}

vector<vector<double>> sum_matrices(
    const vector<vector<double>>& A, 
    const vector<vector<double>>& B) 
{
    int rowsA = A.size(), columnsA = A[0].size();

    if(rowsA != B.size() || columnsA != B[0].size()){
        assert("Incompatible dimensions!");
    }

    vector<vector<double>> C(rowsA, vector<double>(columnsA, 0.0));
    for(int i = 0; i<rowsA; i++){
        for(int j = 0; j<columnsA; j++){
            C[i][j] = A[i][j] + B[i][j];
        }
    }

    return C;
}

// MLP part

class Layer {
    private:
        vector<vector<double>> weights;
        vector<vector<double>> biases;
    
    public:
        Layer(int input_size, int output_size){
            random_device rd; 
            mt19937 gen(rd()); 
            uniform_real_distribution<> dis(0.0, 42.0);

            for (int i = 0; i < input_size; i++) {
                weights[i].resize(input_size); 
                
                for (int j = 0; j < output_size; ++j) {
                    weights[i][j] = dis(gen); 
                }
            }

            for(int i = 0; i< output_size; i++){
                biases[i].resize(output_size);
                for(int j = 0; j<output_size; i++){
                    biases[i][j] = dis(gen);
                }
            }
        }

        vector<vector<double>> forward(vector<vector<double>> input){
            vector<vector<double>> result = dot_product(input, weights);
            result = sum_matrices(result, biases);

            return result;
        }
};

class MLP {
    private:
        int head;
        Layer hidden_layer;
        Layer end_layer;
    
    public:
        MLP(int head_size, int hidden_size, int end_size){
            head = head_size;
            hidden_layer = Layer(hidden_size, end_size);
            end_layer = Layer(end_size,end_size);
        }

        int argmax(vector<double> zs){
            // returns the index of the maximum value of a list
            double max = -1.0;
            int index = 0;
            for(int i = 0; i<zs.size(); i++){
                if(zs[i] > max){
                    max = zs[i];
                    index = i;
                }
            }

            return index;
        }

        // Performs forward propagation
        int forward(vector<vector<double>> input){
            if(input.size() != head){
                cerr << "Mismatch between expected input size (" << head << " ) and what were given (" << input.size() << ")" << endl;
            }

            vector<vector<double>> A1 = dot_product(input, hidden_layer.weights);
            A1 = sum_matrices(A1, hidden_layer.biases);
            A1 = relu(A1);


            vector<vector<double>> A2 = dot_product(A1, end_layer.weights);
            A2 = sum_matrices(A2, end_layer.biases);
            A2 = softmax(A2);

            return argmax(A2);
        }
};



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