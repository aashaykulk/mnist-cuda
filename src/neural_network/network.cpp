/*
 *
 * Multi-Layer Perceptron (MLP) 
 * Input:   [B, 784]
 * Layer 1: [784, 256] + bias → ReLU
 * Layer 2: [256, 128] + bias → ReLU
 * Layer 3: [128, 10]  + bias → logits
 * Loss:    Softmax + Cross-Entropy
 */
#include "network.hpp"
#include <random>
#include <cmath>
#include <iostream>
#include <exception>
#include <stdexcept>
using namespace std;


// Using normal distribution to generate random weights to W1, W2, W3. biases are already 0.
Network::Network() {

  random_device rd;
  mt19937 rng(rd());

  float stddev1 = sqrt(2.0f/in_dim);
  normal_distribution<float> dist(0.0f, stddev1);

  for(float &w: W1) {
    w = dist(rng);
  }

  float stddev2 = sqrt(2.0f/h1);
  normal_distribution<float> dist2(0.0f, stddev2);

  for(float &w: W2) {
    w = dist2(rng);
  }

  float stddev3 = sqrt(2.0f/h2);
  normal_distribution<float> dist3(0.0f, stddev3);

  for(float &w: W3) {
    w = dist3(rng);
  }

  zero_grads();

}

// A = m * k, B = k * n, C = A x B = m * n
void matmul(const vector<float> &A, const vector<float> &B,vector<float> &C, int m, int k, int n) {
  if (A.size() != m*k || B.size() != k*n || C.size() != m*n) {
    throw runtime_error("size mismatch");
    return;
  }
  for (int i = 0; i < m; i++) {
    for (int j  = 0; j < n; j++) {
      float sum = 0.0f;
      for (int l = 0; l < k; l++) {
        sum += A[i*k + l] * B[l*n + j];
      }
      C[i*n + j] = sum;
    }
  }
}

void reLU(vector<float> &z, vector<float> &a) {
  if (a.size() != z.size()) {
    throw runtime_error("reLU size mismatch");

  }
  for (int i = 0; i < z.size(); i++) {
    a[i] = max(0.0f, z[i]);
  }
}

void add_bias(vector<float> &z, const vector<float> &b, int m, int n ) {
  for(int i = 0; i < m; i++) { 
    for (int j = 0; j < n; j++) {
      z[i*n + j] += b[j];
    }
  } 
}

// computes probabilities given logits
void compute_probs(const vector<float> &logits, vector<float> &probs) {

  // compute max logit
  vector<float>::iterator it = max_element(logits.begin(), logits.end());
  float max_logit = *it;

  // compute sum_exp
  float sum_exp = 0.0f;
  for (int j = 0; j < logits.size(); j++) {
    sum_exp += exp(logits[j] - max_logit);
  }

  // compute probs
  for (int j = 0; j < logits.size(); j++) {
    probabilities[j] = (exp(logits[j] - max_logit) / sum_exp);
  }

}
void Network::forward(const vector<float> &X, int B) {
  // make sure batch size is correct
  ensure_batch(B);

  // layer 1 
  matmul(X, W1, z1, B, in_dim, h1);
  add_bias(z1, b1, B, h1);
  reLU(z1, a1);
  
  // layer 2 
  matmul(a1, W2, z2, B, h1, h2);
  add_bias(z2, b2, B, h2);
  reLU(z2, a2);
  
  // layer 3 
  matmul(a2, W3, logits, B, h2, out_dim);
  add_bias(logits, b3, B, out_dim);

  
}

float Network::backward(const vector<float> &X, const vector<uint8_t> &y, int B) {
  compute_probs(logits, probs);

}

void Network::step(float lr) {

}
