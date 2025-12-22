using namespace std;
#include <vector>
#include <string>

class MNIST {
  
  // metadata
  uint32_t num_images;
  uint32_t rows;
  uint32_t cols;

  // data
  vector<uint8_t> images;
  vector<uint8_t> labels;
  
  load(string file);


}
