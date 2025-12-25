/*
 *
 * Multi-Layer Perceptron (MLP) 
 * Input:   [B, 784]
 * Layer 1: [784, 256] + bias → ReLU
 * Layer 2: [256, 128] + bias → ReLU
 * Layer 3: [128, 10]  + bias → logits
 * Loss:    Softmax + Cross-Entropy
 */

#pragma once
#include <vector>
#include <cstdint>
#include <algorithm>

class Network {

  // class constants
  static constexpr int in_dim = 784;
  static constexpr int h1 = 256;
  static constexpr int h2 = 128;
  static constexpr int out_dim = 10;

  int current_B = 0;

  // weights tensors
  std::vector<float> W1 = std::vector<float>(in_dim*h1);
  std::vector<float> W2 = std::vector<float>(h1*h2);
  std::vector<float> W3 = std::vector<float>(h2*out_dim);

  // bias tensors
  std::vector<float> b1 = std::vector<float>(h1, 0.0f);
  std::vector<float> b2 = std::vector<float>(h2, 0.0f);
  std::vector<float> b3 = std::vector<float>(out_dim, 0.0f);

  //intermediate buffers
  std::vector<float> z1;
  std::vector<float> a1;
  std::vector<float> z2;
  std::vector<float> a2;
  std::vector<float> logits;
  std::vector<float> probs;

  // dWeights tensors
  std::vector<float> dW1 = std::vector<float>(in_dim*h1);
  std::vector<float> dW2 = std::vector<float>(h1*h2);
  std::vector<float> dW3 = std::vector<float>(h2*out_dim);
  
  // db tensors
  std::vector<float> db1 = std::vector<float>(h1);
  std::vector<float> db2 = std::vector<float>(h2);
  std::vector<float> db3 = std::vector<float>(out_dim);

  //backprop buffers
  std::vector<float> dlogits;
  std::vector<float> dZ1;
  std::vector<float> dZ2;

 
  // zeroes the gradients after every iteration
  void zero_grads() {

    fill(dW1.begin(), dW1.end(), 0.0f);
    fill(dW2.begin(), dW2.end(), 0.0f);
    fill(dW3.begin(), dW3.end(), 0.0f);

    fill(db1.begin(), db1.end(), 0.0f);
    fill(db2.begin(), db2.end(), 0.0f);
    fill(db3.begin(), db3.end(), 0.0f);

    fill(dZ1.begin(), dZ1.end(), 0.0f);
    fill(dZ2.begin(), dZ2.end(), 0.0f);

    fill(dlogits.begin(), dlogits.end(), 0.0f);
  }

  //batch resize
  void ensure_batch(int B) {

    if (current_B == B) return;

    z1.resize(B*h1);
    z2.resize(B*h2);

    a1.resize(B*h1);
    a2.resize(B*h2);

    logits.resize(B*out_dim);
    probs.resize(B*out_dim);
    dlogits.resize(B*out_dim);

    dZ1.resize(B*h1);
    dZ2.resize(B*h2);
    current_B = B;
  }

  void compute_probs(const std::vector<float> &logits, std::vector<float> &probs);

  public:

  //constructor
  Network();

  //getters
  const std::vector<float>& get_logits() const {return logits;}

  // ML Algos
  void forward(const std::vector<float> &X, int B);
  float backward(const std::vector<float> &X, const std::vector<uint8_t> &y, int B);
  void step(float lr); 

  // training functions
  void save(const std::string &binary);
  void load(const std::string &binary);

};
