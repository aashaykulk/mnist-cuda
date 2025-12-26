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
  MNIST train;
  if (!train.load("../data/mnist/train-images-idx3-ubyte", "../data/mnist/train-labels-idx1-ubyte")) {
    cout << "training load failed\n";
    return 1;
  }
  cout << "Training data loaded.\n";

  Network network;

  int B = 128;
  int N = train.get_num_images();
  int num_batches = N / B;
  int epochs = 7;
  float lr = 0.1f;

  size_t image_size = train.image_size();
  const uint8_t *labels = train.get_labels_buffer();

  // get the images & labels 
  for (int epoch = 0; epoch < epochs; epoch++) {
    float epoch_loss_sum = 0.0f;

    vector<float> X(B*image_size);
    vector<uint8_t> y_batch(B);

    for (int i = 0; i < num_batches; i++) {
      for (int b = 0; b < B; b++) {
        y_batch[b] = labels[i*B + b];
        const uint8_t *img = train.image_ptr(i*B + b);
        for (int p = 0; p < image_size; p++) {
          X[b*image_size + p] = img[p] / 255.0f;
        }
      }
      network.forward(X, B);
      float loss = network.backward(X, y_batch, B);
      network.step(lr);

      epoch_loss_sum += loss;

      if (i % 100 == 0) {
        cout << "epoch " << epoch
             << " batch " << i << "/" << num_batches
             << " loss " << loss << "\n";
      }
    }
    float acc = compute_accuracy(network, train, B);
    cout << "epoch " << epoch
         << " mean loss " << (epoch_loss_sum / num_batches)
         << " accuracy " << (acc)
         << "\n";
  }
  network.save("../models/model.bin");
  return 0;
}
