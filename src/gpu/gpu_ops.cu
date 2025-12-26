#include <vector>
#include "cublas_v2.h"

struct GpuContext {
  
  // device buffers for weights & biases
  float *d_W1, *d_W2, *dW_3;
  float *d_b1, *d_b2, *db_3;

  //device buffers for forward activations
  float *d_X, *d_z1, *d_a1, *d_z2, *d_a2, *d_logits;

  cublasHandle_t handle;
  int maxB;
}
