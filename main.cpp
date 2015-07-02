#include <iostream>
#include <cmath>
#include <iomanip>
using namespace std;

//Parameters
const int maxClusters = 4;
const int vectorLength = 4;
const double decayRate = 0.96;              //About 100 iterations.
const double minAlpha = 0.01;

//Neighbor Updating Parameter
const double radiusReductionPoint = 0.023;  //Last 20% of iterations.
int reductionPoint = 0;
int reductionFlag = 1;

double alpha = 0.6;
double d[maxClusters];                      //Network nodes.

//Debug Flag
int updateNeighbor = 1;
int debug =0 ;

//Weight matrix with randomly chosen values between 0.0 and 1.0
double w[maxClusters][vectorLength] = {{0.2, 0.6, 0.5, 0.1},
                                 {0.9, 0.3, 0.6, 0.4},
                                 {0.3, 0.5, 0.9, 0.2},
                                 {0.4, 0.9, 0.2, 0.3}};

//Training patterns.
const int NUM_TRAINING_PATTERN = 13;
//int training_pattern[NUM_TRAINING_PATTERN][vectorLength] = {
//                                                          {0, 0, 0, 1},
//                                                          {0, 0, 1, 0},
//                                                          {0, 1, 0, 0},
//                                                          {1, 0, 0, 0}
//                                                          };
int training_pattern[NUM_TRAINING_PATTERN][vectorLength] = {{0, 0, 0, 0},
                                                          {0, 0, 0, 1},
                                                          {0, 0, 1, 0},
                                                          {0, 0, 1, 1},
                                                          {0, 1, 0, 0},
                                                          {1, 0, 0, 1},
                                                          {1, 0, 1, 0},
                                                          {1, 0, 1, 1},
                                                          {1, 1, 0, 0},
                                                          {1, 1, 0, 1},
                                                          {1, 1, 1, 0},
                                                          {1, 1, 1, 1}
                                                          };

//Testing patterns to try after training is complete.
const int inputTests = 4;
int tests[inputTests][vectorLength] = {{0, 0, 0, 1},
                                 {1, 0, 0, 0},
                                 {0, 0, 1, 0},
                                 {0, 1, 0, 0}
                                 };
//////////////////////////////////////////////////////////////////////

void initArray(double *d){
    for(int i = 0 ; i < maxClusters ; i++){
        d[i] = 0.0;
    }
}

void distArrayCalculation(int inputArray[NUM_TRAINING_PATTERN][vectorLength] , int vectorSelector){
    initArray(d);
    for(int i = 0; i <maxClusters; i++){ // W[i]
        for(int j = 0; j <vectorLength; j++){ // W[i][j=features in that vector]
            //EUCLIDIAN DISTANCE
                d[i] += pow((w[i][j] - inputArray[vectorSelector][j]), 2);
                if(debug){
                        cout << pow((w[i][j] - inputArray[vectorSelector][j]), 2) << " ";
                }
        }
    }
}

void updateWinnerWeight(int winnerIndex , int vectorSelector){
    for(int i = 0; i < vectorLength ; i++){
        //Update the winner W[winnerIdx][feature]
        //Using OLD_WEIGHT and train_pattern[Vector in that round][features]
        //With Alpha factor
        w[winnerIndex][i] = w[winnerIndex][i] + (alpha * (training_pattern[vectorSelector][i] - w[winnerIndex][i]));

        //UPDATE NEIGHTBOR VALUE (DEFAULT THIS CODE IS MUTE BY updateNeighbor FLAG)
        if(alpha > radiusReductionPoint && updateNeighbor){
            if((winnerIndex > 0) && (winnerIndex < (maxClusters - 1))){ //NEIGHBOR IN RANGE
                //Update neighbor to the left...
                w[winnerIndex - 1][i] = w[winnerIndex - 1][i] +
                    (alpha * (training_pattern[vectorSelector][i] - w[winnerIndex - 1][i]));
                //and update neighbor to the right.
                w[winnerIndex + 1][i] = w[winnerIndex + 1][i] +
                    (alpha * (training_pattern[vectorSelector][i] - w[winnerIndex + 1][i]));
            } else {
                if(winnerIndex == 0){ //Neighbor is on LEFT EDGE
                    //Update neighbor to the right.
                    w[winnerIndex + 1][i] = w[winnerIndex + 1][i] +
                        (alpha * (training_pattern[vectorSelector][i] - w[winnerIndex + 1][i]));
                } else { //Neighbor is on RIGHT EDGE
                    //Update neighbor to the left.
                    w[winnerIndex - 1][i] = w[winnerIndex - 1][i] +
                        (alpha * (training_pattern[vectorSelector][i] - w[winnerIndex - 1][i]));
                }
            }
        }
    }
}

int findMinimumIdx(double *tempArray){
    int tempValue = 65535;
    int tempIndex = -1;
    for(int i = 0 ; i < maxClusters ; i++){
        if(tempArray[i] < tempValue){
                tempIndex = i;
                tempValue = tempArray[i];
        }
    }
    return tempIndex;
}

void training(){
    int iteration = 0;
    int distArrayMinIdx = 0;

    do{
        //SELECT INPUT PATTERNS
        for( int trainVector_selector = 0 ; trainVector_selector < NUM_TRAINING_PATTERN ; trainVector_selector++ ){
            // find d[] =>CALCULATE DISTANCE PATTERN <=> NODE_NET
            distArrayCalculation(training_pattern , trainVector_selector);
            // got d[] FIND WINNER IN THIS ROUND
            distArrayMinIdx = findMinimumIdx(d);
            //UPDATE WEIGHT ON WINNER ( and it's neighbor)
            updateWinnerWeight(distArrayMinIdx,trainVector_selector);
        }
        //LEARNING RATE DECREASE EVERY ROUND (same as human)
        alpha = decayRate * alpha;
        //UPDATE ITERATOR
        iteration++;
        //UPDATE NEIGHBOR UPDATOR Parameter
        if(alpha < radiusReductionPoint && reductionFlag && updateNeighbor){
                reductionFlag = 0;
                reductionPoint = iteration;
        }

    }while(alpha > minAlpha);

    cout << "=========================" << endl;
    cout << " END OF TRANING PROCESS  " << endl;
    cout << "=========================" << endl;
    cout << "  - total iteration round = " << iteration <<endl;
    cout << "=========================" << endl;

}

void displayArray(double w[maxClusters][vectorLength]){
    for(int i = 0 ; i < maxClusters ; i++ ){
            cout << "[" <<i<<"]  " ;
        for(int j = 0 ; j < vectorLength ; j++){
            cout << w[i][j] << ((j==vectorLength-1)? "":" , ");
        }
        cout <<endl;
    }
}

void classify_training(){
    int distArrayMinIdx=-1;
    cout << "Training Vectors : " <<endl;
    for( int trainVector_selector = 0 ; trainVector_selector < NUM_TRAINING_PATTERN ; trainVector_selector++ ){
            // find d[] =>CALCULATE DISTANCE PATTERN <=> NODE_NET
            distArrayCalculation(training_pattern , trainVector_selector);
            // got d[] FIND WINNER IN THIS ROUND
            distArrayMinIdx = findMinimumIdx(d);

            cout << "<";
            for(int i = 0; i < vectorLength; i++){
                cout << training_pattern[trainVector_selector][i] << ((i==vectorLength-1)? "":" , ");
            }
            cout << "> fits into category " << distArrayMinIdx << endl;
    }
}

void classify_testing(){
    int distArrayMinIdx=-1;
    cout << "Testing Vectors : " <<endl;
    for( int testVector_selector = 0 ; testVector_selector < inputTests ; testVector_selector++ ){
            // find d[] =>CALCULATE DISTANCE PATTERN <=> NODE_NET
            distArrayCalculation(tests , testVector_selector);
            // got d[] FIND WINNER IN THIS ROUND
            distArrayMinIdx = findMinimumIdx(d);
            cout << "<";
            for(int i = 0; i < vectorLength; i++){
                cout << tests[testVector_selector][i] << ((i==vectorLength-1)? "":" , ");
            }
            cout << "> fits into category " << distArrayMinIdx << endl;
    }
}

int main()
{
    cout << fixed << setprecision(3) << endl;  //Format all the output.
    cout << "SELF-ORGANIZING MAP VER 0.1 !" << endl <<endl <<endl;

    training();

    cout << "=========================" << endl;
    cout << " Weight Array            " << endl;
    cout << "=========================" << endl;
        displayArray(w);
    cout << "=========================" << endl;

    //TESTING ON TRAINING DATA
    cout << "=========================" << endl;
    cout << " Testing On Training Data" << endl;
    cout << "=========================" << endl;
        classify_training();
    cout << "=========================" << endl;


    //TESTING ON UNSEEN DATA
    cout << "=========================" << endl;
    cout << " Testing On UNSEEN Data  " << endl;
    cout << "=========================" << endl;
        classify_testing();
    cout << "=========================" << endl;

    return 0;
}
