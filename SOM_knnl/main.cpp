#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <iterator>
#include <string>
#include <array>
#include <neural_net_headers.hpp>
#include <print_network.hpp>
#include <map>
#include <set>

//Plot Stuff
#include <SFML/System.hpp>
#include <SFML/Graphics.hpp>

using namespace std;

unsigned int ROW = 40;
unsigned int COL = 40;
int line_count = 0;
int element_count =0;
//DATA STRUCTURE FOR Data in form of VECTORs
typedef std::vector <double> vector_data;
typedef std::vector <vector_data> vector_container;

//CREATE DATASET
    vector<vector<double> > data;

//Activation Function
typedef neural_net::Cauchy_function < vector_data::value_type, vector_data::value_type, int > C_function_type;
C_function_type cauchy_function(2.0,1);

//Distance Unit
typedef distance::Euclidean_distance_function<vector_data> Euclid_distance_type;
Euclid_distance_type euclid_distance;

//Newron Typedef
typedef neural_net::Basic_neuron <C_function_type,Euclid_distance_type> Kohonen_neuron_type;

//Network Build Rectangular Topology
typedef neural_net::Rectangular_container < Kohonen_neuron_type > Kohonen_network;
Kohonen_network kohonen_network;

template<typename T, size_t N>
vector<T> convert_array_to_vector(const T (&source_array)[N]) {
    return vector<T>(source_array, source_array+N);
}

void readfile_ucl(){
    string line,value;
    ifstream inputfile("iris.data");

    //Counting The Inputs and Elements (Last is Class Tag)
    if (inputfile.is_open()){
        //Get First Line To Count Element
        string temp;
        getline ( inputfile, temp );
        for(unsigned int i = 0 ; i < temp.size() ; i++){
            if(temp[i] == ',')element_count++;
        }
        cout << " ELEMENT COUNT = " << element_count <<endl;
        //count the first line
        line_count++;
        while(getline ( inputfile, line )){
            line_count++;
        }
        cout << " LINE COUNT = " << line_count <<endl;
    }
    inputfile.close();
    inputfile.clear();

    //Input Gathering From File
    inputfile.open("iris.data");
    if (inputfile.is_open()){
    unsigned counter = line_count;
        while ( --counter !=0  ){
            getline ( inputfile, value , ',');
            vector<double> one_input;
            double d = strtod(value.c_str(), NULL);
            one_input.push_back(d);
            for( int i = 1 ; i< element_count ;i++){
                getline ( inputfile, value , ',');
                d = strtod(value.c_str(), NULL);
                one_input.push_back(d);
            }
            //DROP TAG
            getline ( inputfile, value , '\n');

//            cout << "VECTOR CONTENT = <" ;
//            for ( vector<double>::iterator i = one_input.begin() ; i < one_input.end() ; i ++){
//                cout << *i << ((i==one_input.end()-1)? "":",");
//            }
//            cout << ">" <<endl;
            data.push_back(one_input);
        }
    inputfile.close();
    }
}

vector<pair<double,double> > findMinMax (){
    vector<pair<double,double> > results;
    double minimum = 10000;
    double maximum = -10000;
    for(unsigned int i = 0 ; i < element_count ; i++ ){
        for(unsigned int j = 0 ; j < line_count-1 ; j++){
            if(data[j][i] < minimum) minimum = data[j][i];
            if(data[j][i] > maximum) maximum = data[j][i];
        }
        results.push_back(make_pair(minimum,maximum));
    }
    return results;
}

void normalization(){
    vector<pair<double,double> > max_min = findMinMax();
    double data_min[element_count];
    double data_max[element_count];
    for(unsigned int i = 0 ; i < element_count ; i++){
        data_max[i] = max_min[i].second;
        data_min[i] = max_min[i].first;
    }
    for(int attribute = 0 ; attribute < element_count ; attribute++){
        for(int index =0 ; index < line_count-1 ; index++){
            data[index][attribute] = (data[index][attribute] - data_min[attribute]) / (data_max[attribute] - data_min[attribute]);
        }
    }
}

void printVector(vector_container vec){

    vector_container::iterator row;
    vector_data::iterator col;

    for(row = vec.begin() ; row <vec.end() ; row++){
        for(col = row->begin() ; col < row->end() ; col++ ){
            cout << *col << ((col == row->end()-1)? "": ", ") ;
        }
        cout << endl;
    }
}

void printWeight(){
        for(unsigned int i = 0 ; i < ROW ; i++){
            for(unsigned int j = 0 ;j < COL ; j++){
                cout << "DETAIL FOR WEIGHT["<<i<<"]["<<j<<"] = " <<endl;
                for(vector<double>::iterator it = kohonen_network.objects[i][j].weights.begin() ; it < kohonen_network.objects[i][j].weights.end() ; it++ ){
                    cout << *(it) << "\t";
                } cout <<endl;
            }
            cout <<endl;
        }
}

void text_plotter(){
    ofstream myfile;
    myfile.open("output2.dat");
    for(unsigned int i = 0 ; i < ROW ; i++){
        for(unsigned int j = 0 ; j < COL ; j++){
            myfile << i << ',' << j << ',';
            for(vector<double>::iterator it = kohonen_network.objects[i][j].weights.begin() ; it < kohonen_network.objects[i][j].weights.end() ; it++ ){
                    myfile << *(it) << ((it == kohonen_network.objects[i][j].weights.end()-1)? '\0':',');
            } myfile << '\n';
        }
    }
    myfile.close();

}

int main(){

    cout << "Reading Files" << endl;
    readfile_ucl();
    cout << "Import file successfully" <<endl;
    cout << "Normalizing Data" <<endl;
    normalization();
    cout << "Data is normalized successfully" <<endl;

//    vector_container data =   {{1,0,0,0},
//                                {0,1,0,0},
//                                {0,0,1,0},
//                                {0,0,0,1}};

    neural_net::External_randomize ER;
    neural_net::generate_kohonen_network ( ROW, COL, cauchy_function, euclid_distance, data, kohonen_network, ER );

//    //Implementation Of Winner takes All Algorithm
//    typedef neural_net::Wta_proportional_training_functional < vector_data, double, int > Wta_train_func;
//    Wta_train_func wta_train_func ( 0.1, 0 );
//    //
//    typedef neural_net::Wta_training_algorithm< Kohonen_network,vector_data,
//                                                vector_container::iterator,Wta_train_func> Learning_algorithm;
//    //Apply The Learning Algorithm
//    Learning_algorithm training_alg ( wta_train_func );

    //Implementation Of Winner takes Most Algorithm
//    typedef neural_net::City_topology < size_t > Max_top;
//    Max_top max_top;
    typedef neural_net::Hexagonal_topology < size_t > Max_top;
    Max_top max_top ( kohonen_network.get_no_rows() );

    typedef neural_net::Gauss_function < vector_data::value_type, vector_data::value_type, int > Gauss_function_space;
    Gauss_function_space gauss_function_space ( 100, 1 );

    typedef neural_net::Gauss_function < size_t, vector_data::value_type, int > Gauss_function_net;
    Gauss_function_net gauss_function_net ( 10, 1 );

    typedef neural_net::Classic_training_weight<std::vector<double>, int ,Gauss_function_net, Gauss_function_space,Max_top,Euclid_distance_type,size_t >Classic_weight;
    Classic_weight classic_weight(gauss_function_net,gauss_function_space,max_top,euclid_distance);

    typedef neural_net::Wtm_classical_training_functional
    <
      vector_data,
      double,
      int,
      size_t,
      Classic_weight
    > Wtm_c_l_f;

    Wtm_c_l_f wtm_c_l_f ( classic_weight, 0.3 ); //0.3

    typedef neural_net::Wtm_training_algorithm
    <
      Kohonen_network,
      vector_data,
      vector_container::iterator,
      Wtm_c_l_f,
      size_t
    > Wtm_c_training_alg;

    Wtm_c_training_alg training_alg ( wtm_c_l_f );


    //Learning Iteration
    cout << "Learning" <<endl;
////    for ( int i = 0; i < 250; ++i )
////    {
////    training_alg ( data.begin(), data.end(), &kohonen_network );
////
////   //decrease sigma parameter in network
////   //will make training process more sharpen with each epoch,
////   //but it have to be done slowly :-)
////    training_alg.training_functional.generalized_training_weight.network_function.sigma
////      *= 2.0/3.0;
////
////   //shuffle data
////    std::random_shuffle ( data.begin(), data.end() );
////    }
    int sizeMonitor = 400;
    sf::RenderWindow window(sf::VideoMode(sizeMonitor, sizeMonitor), "PLOTTER");
    int margin = sizeMonitor/ROW;
    int it = 0;
      while (window.isOpen()){
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        //window.clear();

        if(it < 20){
          training_alg ( data.begin(), data.end(), &kohonen_network );
          std::random_shuffle ( data.begin(), data.end() );
          if(it%10 == 0)cout << "training round = " << it <<endl;

            for(unsigned int i = 0 ; i < ROW ; i++){
                for(unsigned int j = 0 ;j < COL ; j++){
                    sf::RectangleShape temp(sf::Vector2f(margin, margin));
                    temp.setPosition( j * margin , i * margin);
                    int r = 255*kohonen_network.objects[i][j].weights[0];
                    int g = 255*kohonen_network.objects[i][j].weights[1];
                    int b = 255*kohonen_network.objects[i][j].weights[2];
                    int a = 255*3*kohonen_network.objects[i][j].weights[3];
                    sf::Color attribute_color(r,g,b);
                    temp.setFillColor(attribute_color);
                    window.draw(temp);
                }
            }
            it++;
            training_alg.training_functional.generalized_training_weight.network_function.sigma
            *= 2.0/3.0;

        }else{

//        sf::RectangleShape tutor(sf::Vector2f(100,100));
//        tutor.setPosition(200,200);
//        sf::Color c(200,200,200);
//        tutor.setFillColor(c);
//        window.draw(tutor);
//        window.display();
        break;

        }
        window.display();






    }

    cout <<endl << "Learning Phrase Completed ! " <<endl;
    cout << "===============================" <<endl;
//
//sf::RenderWindow window(sf::VideoMode(400, 400), "PLOTTER");
//    int margin = 20;
//    int it = 0;
//    int test;
//      while (window.isOpen()){
//        sf::Event event;
//        while (window.pollEvent(event))
//        {
//            if (event.type == sf::Event::Closed)
//                window.close();
//        }
//
//        window.clear();
//
//        for(int i = 0 ; i < ROW ; i++){
//                for(int j = 0 ;j < COL ; j++){
//                    sf::RectangleShape temp(sf::Vector2f(20, 20));
//                    temp.setPosition(i*margin , j * margin);
//                    int r = 255*kohonen_network.objects[i][j].weights[0];
//                    int g = 255*kohonen_network.objects[i][j].weights[1];
//                    int b = 255*kohonen_network.objects[i][j].weights[2];
//                    int a = 255*kohonen_network.objects[i][j].weights[3];
//                    sf::Color attribute_color(r,g,b,a);
//                    temp.setFillColor(attribute_color);
//                    window.draw(temp);
//                }
//        }
//
//        window.display();
//      }
        cout << setprecision(9);
        cout << fixed;
        //neural_net::print_network(cout,kohonen_network,test_vector);
        cout << endl << endl;

        int min_x=0;
        int min_y=0;
        double min_d = 65536;

        map<pair<int,int> , int > myMap;
        set<pair<int,int> > differ;
        for(int it = 0 ; it < 150 ; it++){
            for(int i = 0 ; i < ROW ; i++ ){
                for(int j = 0 ; j < COL ; j++){
                    double temp = kohonen_network(i,j)(data[it]);
                    if(temp < min_d){
                        min_x = i;
                        min_y = j;
                        min_d = temp;
                    }
                }
            }
            cout << " MIN (X,Y) = (" << min_x << "," << min_y << ")" <<endl;
            cout << " WITH DISTANCE = " << min_d <<endl;
            differ.insert(make_pair(min_x,min_y));
            myMap[make_pair(min_x,min_y)] +=1;

            sf::RectangleShape tempa(sf::Vector2f(margin, margin));
            tempa.setPosition( min_x * margin , min_y * margin);
            sf::Color attribute_color(255,255,255);
            tempa.setFillColor(attribute_color);
            window.draw(tempa);
            window.display();


            min_x = 0;
            min_y = 0;
            min_d = 65536;
        }
//        cout << "found" << endl;
//        for(set<pair<int,int>,int>::iterator it = differ.begin() ; it != differ.end() ; it++ ){
//            cout << it->first << "," << it->second << " = " << myMap[make_pair(it->first,it->second)] <<endl;
//        }

    //text_plotter();

    cout << "END OF PLOTTING PROCESS" << endl;

    cin.ignore();


    return 0;
}
