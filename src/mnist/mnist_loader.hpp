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
  uint32_t get_num_images() const;
  uint32_t get_num_labels() const;
  uint32_t get_num_rows() const;
  uint32_t get_num_cols() const;
  const uint8_t *get_images_buffer() const;
  const uint8_t *get_labels_buffer() const;
  const uint8_t *image_ptr(size_t i) const;
  
  // bytes per image
  size_t image_size() const;
  // load image & label file
  bool load(const std::string &image_file, const std::string &label_file);



};
