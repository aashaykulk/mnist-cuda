#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <cstddef>

class MNIST {
  
  // metadata
  uint32_t num_images = 0;
  uint32_t num_labels = 0;
  uint32_t num_rows = 0;
  uint32_t num_cols = 0;

  // data
  std::vector<uint8_t> images;
  std::vector<uint8_t> labels;
 
  public:

  // getters
  uint32_t get_num_images() const {return num_images;}
  uint32_t get_num_labels() const {return num_labels;}
  uint32_t get_num_rows() const {return num_rows;}
  uint32_t get_num_cols() const {return num_cols;}
  const uint8_t *get_images_buffer() const {return images.data();}
  const uint8_t *get_labels_buffer() const {return labels.data();}
  const uint8_t *image_ptr(size_t i) const;
  // bytes per image
  size_t image_size() const {return num_rows * num_cols;}

  // load image & label file
  bool load(const std::string &image_file, const std::string &label_file);



};
