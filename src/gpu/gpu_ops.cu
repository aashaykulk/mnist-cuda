#include <vector>
#include <cuda_runtime.h>
#include "cublas_v2.h"

static constexpr int in_dim = 784;
static constexpr int h1 = 256;
static constexpr int h2 = 128;
static constexpr int out_dim = 10;

// GpuContext is a host side context containing pointers to device side buffers
struct GpuContext {
  
  // device buffers for weights & biases
  float *d_W1, *d_W2, *d_W3;
  float *d_b1, *d_b2, *d_b3;

  //device buffers for forward activations
  float *d_X, *d_z1, *d_a1, *d_z2, *d_a2, *d_logits;

  cublasHandle_t handle;
  int maxB;

  GpuContext() {
    d_W1 = nullptr;
    d_W2 = nullptr;
    d_W3 = nullptr;

    d_b1 = nullptr;
    d_b2 = nullptr;
    d_b3 = nullptr;

    d_X = nullptr;
    d_z1 = nullptr;
    d_a1 = nullptr;
    d_z2= nullptr;
    d_a2 = nullptr;
    d_logits = nullptr;
    handle = CUBLAS_STATUS_INVALID_VALUE;
    maxB = 0;
  }
};


GpuContext *gpu_create(int maxB) {
  GpuContext *ctx = new GpuContext;
  ctx->maxB = maxB;
  
  cublasStatus_t stat;
  stat = cublasCreate(&ctx->handle);
  //malloc for all device buffers
  cudaMalloc(&ctx->d_W1, in_dim*h1*sizeof(float));
  cudaMalloc(&ctx->d_W2, h1*h2*sizeof(float));
  cudaMalloc(&ctx->d_W3, h2*out_dim*sizeof(float));

  cudaMalloc(&ctx->d_b1, h1*sizeof(float));
  cudaMalloc(&ctx->d_b2, h2*sizeof(float));
  cudaMalloc(&ctx->d_b3, out_dim*sizeof(float));

  cudaMalloc(&ctx->d_X, maxB*in_dim*sizeof(float));
  cudaMalloc(&ctx->d_z1, maxB*h1*sizeof(float));
  cudaMalloc(&ctx->d_a1, maxB*h1*sizeof(float));
  cudaMalloc(&ctx->d_z2, maxB*h2*sizeof(float));
  cudaMalloc(&ctx->d_a2, maxB*h2*sizeof(float));
  cudaMalloc(&ctx->d_logits, maxB*out_dim*sizeof(float));

  return ctx;
}
void gpu_destroy(GpuContext *ctx) {
  if (!ctx) return;

}
