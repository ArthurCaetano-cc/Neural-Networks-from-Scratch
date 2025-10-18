/* 
How to compile: 
    g++ Utility.cpp Activations.cpp Linear.cpp Dataset.cpp mlp.cpp -o main

How to execute:
./main ./Datasets/digit-recognizer/train.csv ./Dat
asets/digit-recognizer/test.csv

Obs:
    Utility has problems with templates, please add template to input
    matrices (output will be always vector<vector<double>>)
*/

#include "Activations.h"
#include "Utility.h"
#include "Linear.hpp"
#include "Dataset.hpp"
#include <cfloat>

using namespace std;

// Hyperparameters:
int NUM_CLASSES = 10, IMG_SIZE=784;

class MLP {
private:
    LinearLayer<uint32_t> inputLayer;
    LinearLayer<double> hiddenLayer, outputLayer;
public:
    //constructor
    MLP(int in_f=IMG_SIZE, int hid=10, int out_f=NUM_CLASSES)
        : inputLayer(in_f, hid, Activation::ReLU),
          hiddenLayer(hid, hid, Activation::ReLU),
          outputLayer(hid, out_f, Activation::Softmax)
    {}

    vector<double> format_softmax(vector<vector<double>>softmax_output);

    int arg_max(vector<double>output){
        double max = DBL_MIN, maxIndex = 0;
        for(int i = 0; i<output.size(); i++){
            if(output[i] > max){
                max = output[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }


    // returns the predicted label
    int forward(vector<vector<uint32_t>> img){
        vector<vector<double>> x = inputLayer.forward(img);
        x = hiddenLayer.forward(x);
        x = outputLayer.forward(x);

        printMatrix(x);
        return 0;
    }
};

int main(int argc, char* argv[]){

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

    MLP mlp = MLP();
    cout << "Multilayer Perceptron initialized" << endl;

    return 0;
}

/*              
    (1, 784) -> input -> (1, 10) -> Hidden -> (1, 10) -> output -> (1, 10)
*/