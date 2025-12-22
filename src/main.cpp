using namespace std;
#include <iostream>
#include "mnist/mnist_loader.hpp"

int main() {
  MNIST train;
  MNIST test;
  if (!train.load("../data/mnist/train-images-idx3-ubyte", "../data/mnist/train-labels-idx1-ubyte")) {
    cout << "training load failed\n";
    return 1;
  }
  if (!test.load("../data/mnist/t10k-images-idx3-ubyte", "../data/mnist/t10k-labels-idx1-ubyte")) {
    cout << "test load failed\n";
    return 1;
  }


  cout << "test load succeeded\n";
  return 0;
}

