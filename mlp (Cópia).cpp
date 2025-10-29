/* 
How to compile: 
    g++ Utility.cpp Activations.cpp Linear.cpp Dataset.cpp mlp.cpp -o main

How to execute:
./main ./Datasets/digit-recognizer/train.csv ./Datasets/digit-recognizer/test.csv

Obs:
    Utility has problems with templates, please add template to input
    matrices (output will be always vector<vector<double>>)
*/

#include "Utility.h" /* Use for debugging matrices (printMatrix)*/
#include "Activations.h"
#include "Linear.hpp"
#include "Dataset.hpp"
#include <cfloat>
#include <cstdint>
#include <vector>

using namespace std;

// Hyperparameters:
int NUM_CLASSES = 10, IMG_SIZE=784, EPOCHS=10;
double LR=0.001;

class MLP {
public: //should be private
    LinearLayer inputLayer, hiddenLayer, outputLayer;
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
    vector<vector<double>> forward(vector<vector<uint32_t>> img){
        vector<vector<double>> x = inputLayer.forward<uint32_t>(img);
        x = hiddenLayer.forward<double>(x);
        x = outputLayer.forward<double>(x);

        return x;
    }

    void optimizer_step(vector<vector<uint32_t>> input, vector<vector<double>> output, 
        vector<vector<double>> expectedOutput){

                // gradient computing:

                //outputLayer:
                vector<vector<double>> delta3 = sub_matrices(output, expectedOutput);
                vector<vector<double>> dW3 = dot_product(transpoose(hiddenLayer.a), delta3),
                    dB3 = delta3;

                
                // hidden layer gradient:
                vector<vector<double>> delta2 = dot_product(delta3, transpoose(outputLayer.weights));
                delta2 = hadamard_product(delta2, ReLU_derivative(hiddenLayer.z));


                vector<vector<double>> dW2 = dot_product(transpoose(inputLayer.a), delta2),
                                        dB2 = delta2;
                
                // input layer gradient:
                vector<vector<double>> delta1 = dot_product(delta2, transpoose(hiddenLayer.weights));
                delta1 = hadamard_product(delta1, ReLU_derivative(inputLayer.z));


                vector<vector<double>> dW1 = dot_product(transpoose(input), delta1),
                                        dB1 = delta1;

                // SGD optimization:
                
                // output
                outputLayer.weights = sub_matrices(outputLayer.weights, constantMult(dW3, LR));
                outputLayer.biases = sub_matrices({outputLayer.biases}, constantMult(dB3, LR))[0];
            
                // hidden layer:
                hiddenLayer.weights = sub_matrices(hiddenLayer.weights, constantMult(dW2, LR));
                hiddenLayer.biases = sub_matrices({hiddenLayer.biases}, constantMult(dB2, LR))[0];

                // input layer:
                inputLayer.weights = sub_matrices(inputLayer.weights, constantMult(dW1, LR));
                inputLayer.biases = sub_matrices({inputLayer.biases}, constantMult(dB1, LR))[0];
            
            }

};

int from_one_hot(vector<double>label){
    for(int i = 0; i<label.size(); i++){
        if(label[i] == 1){
            return i;
        } 
    }

    return -1;
}

int main(int argc, char* argv[]){
    if(argc < 2){
        cerr << "Please provide the train.csv" << endl;
        return 1;
    }
    
    cout << "[INFO] Reading files..." << endl;
    Dataset train =  load_csv(argv[1]);

    cout << "Reading completed:" << endl;
    cout << "Length of train.csv: " << train.n << endl;

    MLP mlp = MLP();
    cout << "Multilayer Perceptron initialized" << endl;

    cout << "Starting training" << endl;
    /* Training: */
    for(int i = 0; i<EPOCHS; i++){
        cout << "Epoch: " << i+1 << "/" << EPOCHS << endl;
        
        vector<int> predictions(train.n);
        vector<int> expected(train.n);

        for(int j = 0; j<train.n; j++){
            if (j == 0 && i == 0) {
                cout << "First output weight before: " << mlp.outputLayer.weights[0][0] << endl;
                cout << "First hidden weight before: " << mlp.hiddenLayer.weights[0][0] << endl;
                cout << "First input weight before: " << mlp.inputLayer.weights[0][0] << endl;
            }

            if (j == 1) {
                cout << "\nAfter first update: " << mlp.outputLayer.weights[0][0] << " " << mlp.hiddenLayer.weights[0][0] << " " << mlp.inputLayer.weights[0][0] << endl;
            }

            vector<vector<double>> y_hat = mlp.forward({train.images[j]});
            mlp.optimizer_step({train.images[j]}, y_hat, {train.labels[j]});
            
            predictions[j] = mlp.arg_max(y_hat[0]);
            expected[j] = from_one_hot(train.labels[j]);
        }

        cout << "Accuracy: " << accuracy(predictions, expected) << endl;
    }



    return 0;
}

/*              
    (1, 784) -> input -> (1, 10) -> Hidden -> (1, 10) -> output -> (1, 10)
*/