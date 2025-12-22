using namespace std;
#include "mnist_loader.hpp"
#include <fstream>
#include <iostream>

bool read_u32_be(istream &ifs, uint32_t &u32) {
  unsigned char buf[4];
  ifs.read(reinterpret_cast<char *>(buf), 4);
  if (!ifs) return false; 
  u32 = ((uint32_t)buf[0] << 24) + ((uint32_t)buf[1] << 16) + ((uint32_t)buf[2] << 8) + (uint32_t) buf[3];
  return true;
}

size_t MNIST::image_size() const {
  return num_rows * num_cols;
}

bool MNIST::load(const std::string &image_file, const std::string &label_file) {
  ifstream image_stream;
  ifstream label_stream;

  image_stream.open(image_file, std::ios::binary);
  label_stream.open(label_file, std::ios::binary);

  if (!image_stream.is_open() || !label_stream.is_open()) {
    cout << "Error opening image or label\n";
    return false;
  }

  uint32_t magic_image = 0;
  if (!read_u32_be(image_stream, magic_image)) {
    return false;
  }

  uint32_t magic_label= 0;
  if (!read_u32_be(label_stream, magic_label)) {
    return false;
  }

  if (magic_image != 2051 || magic_label != 2049) {
    cout << "Incorrect magic number for image or label\n";
    return false;
  }

  if (!read_u32_be(image_stream, num_images) || !read_u32_be(label_stream, num_labels)) {
    return false;
  }

  if (num_labels != num_images) {
    cout << "num_images and num_labels don't match\n";
    return false;
  }

  if (!read_u32_be(image_stream, num_rows) || !read_u32_be(image_stream,  num_cols)) {
    return false;
  }

  images.resize(num_images * num_rows * num_cols);
  labels.resize(num_labels);

  image_stream.read(reinterpret_cast<char *>(images.data()), images.size());
  label_stream.read(reinterpret_cast<char *>(labels.data()), labels.size());

  if (!image_stream || !label_stream) {
    cout << "data load failed\n"; return false;
  }

  return true;
}

