/*
 *
* Input:   [B, 784]
* Layer 1: [784, 256] + bias → ReLU
* Layer 2: [256, 128] + bias → ReLU
* Layer 3: [128, 10]  + bias → logits
* Loss:    Softmax + Cross-Entropy
*/

#pragma once
#include <vector>
#include <cstdint>

class Network {

  // class constants
  static constexpr int in_dim = 784;
  static constexpr int h1 = 256;
  static constexpr int h2 = 128;
  static constexpr int out_dim = 10;

  // weights tensors
  std::vector<float> W1;
  std::vector<float> W2;
  std::vector<float> W3;

  // bias tensors
  std::vector<float> b1;
  std::vector<float> b2;
  std::vector<float> b3;

  //intermediate buffers
  std::vector<float> z1;
  std::vector<float> a1;
  std::vector<float> z2;
  std::vector<float> a2;
  std::vector<float> logits;
  std::vector<float> probs;

  // dWeights tensors
  std::vector<float> dW1;
  std::vector<float> dW2;
  std::vector<float> dW3;
  
  // db tensors
  std::vector<float> db1;
  std::vector<float> db2;
  std::vector<float> db3;

  //backprop buffers
  std::vector<float> dlogits;
  std::vector<float> dZ1;
  std::vector<float> dZ2;

  int current_B = 0;
  
  void zero_grads();

  //batch resize
  void ensure_batch(int B);

  public:

  //constructor
  Network();

  //getters
  const std::vector<float>& get_logits() const {return logits;}

  // ML Algos
  void forward(const std::vector<float> &X, int B);
  float backward(const std::vector<float> &X, const std::vector<uint8_t> &y, int B);
  void step(float lr); 

};
