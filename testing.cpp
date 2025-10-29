#include "Utility.h"
#include <vector>

using namespace std;

int main(){
    vector<double> output = {1, 2, 3, 4, 5, 6};
    vector<double> expected = {6, 5, 3, 3, 2, 6};
   
    printMatrix(sub_matrices({expected}, {output}));

    return 0;
}