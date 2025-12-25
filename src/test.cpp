#include <iostream>
#include "mnist/mnist_loader.hpp"
#include "neural_network/network.hpp"
#include <vector>
using namespace std;

static float compute_accuracy(Network &net, const MNIST &ds, int B) {
  const int N = static_cast<int>(ds.get_num_images());
  const int image_size = static_cast<int>(ds.image_size());
  const uint8_t *labels = ds.get_labels_buffer();

  const int num_batches = N / B;

  std::vector<float> X(B * image_size);

  int correct = 0;
  int total = 0;

  for (int i = 0; i < num_batches; i++) {
    // fill X for this batch
    for (int b = 0; b < B; b++) {
      const int idx = i * B + b;
      const uint8_t *img = ds.image_ptr(idx);

      for (int p = 0; p < image_size; p++) {
        X[b * image_size + p] = img[p] / 255.0f;
      }
    }

    // forward pass
    net.forward(X, B);

    // read logits and compute argmax per sample
    const std::vector<float> &logits = net.get_logits(); // size B*10

    for (int b = 0; b < B; b++) {
      int pred = 0;
      float best = logits[b * 10 + 0];

      for (int c = 1; c < 10; c++) {
        float v = logits[b * 10 + c];
        if (v > best) {
          best = v;
          pred = c;
        }
      }

      const int idx = i * B + b;
      if (static_cast<uint8_t>(pred) == labels[idx]) correct++;
      total++;
    }
  }

  return (total > 0) ? (static_cast<float>(correct) / total) : 0.0f;
}

int main() {
  MNIST test;
  if (!test.load("../data/mnist/t10k-images-idx3-ubyte", "../data/mnist/t10k-labels-idx1-ubyte")) {
    cout << "testing load failed\n";
    return 1;
  }
  cout << "test data loaded.\n";

  Network network;
  network.load("../models/model.bin");

  int B = 256; 
  float acc = compute_accuracy(network, test, B);
  cout << " accuracy " << (acc)
       << "\n";
  return 0;
}
