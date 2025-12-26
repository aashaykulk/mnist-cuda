#include <iostream>
#include <cstdlib>
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
  bool handle_created;
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

    handle = NULL;
    handle_created = false;

    maxB = 0;
  }
};


void gpu_destroy(GpuContext *ctx) {
  if (!ctx) return;

  if (ctx->d_W1) {cudaFree(ctx->d_W1); ctx->d_W1 = nullptr;}
  if (ctx->d_W2) {cudaFree(ctx->d_W2); ctx->d_W2 = nullptr;}
  if (ctx->d_W3) {cudaFree(ctx->d_W3); ctx->d_W3 = nullptr;}

  if (ctx->d_b1) {cudaFree(ctx->d_b1); ctx->d_b1 = nullptr;}
  if (ctx->d_b2) {cudaFree(ctx->d_b2); ctx->d_b2 = nullptr;}
  if (ctx->d_b3) {cudaFree(ctx->d_b3); ctx->d_b3 = nullptr;}

  if (ctx->d_X) {cudaFree(ctx->d_X); ctx->d_X = nullptr;}
  if (ctx->d_z1) {cudaFree(ctx->d_z1); ctx->d_z1 = nullptr;}
  if (ctx->d_a1) {cudaFree(ctx->d_a1); ctx->d_a1 = nullptr;}
  if (ctx->d_z2) {cudaFree(ctx->d_z2); ctx->d_z2 = nullptr;}
  if (ctx->d_a2) {cudaFree(ctx->d_a2); ctx->d_a2 = nullptr;}
  if (ctx->d_logits) {cudaFree(ctx->d_logits); ctx->d_logits = nullptr;}

  if (ctx->handle_created) cublasDestroy(ctx->handle);
  maxB = 0;

  delete ctx;


}

GpuContext *gpu_create(int maxB) {
  GpuContext *ctx = new GpuContext;
  ctx->maxB = maxB;
  
  cublasStatus_t stat;
  stat = cublasCreate(&ctx->handle);
  if (stat != CUBLAS_STATUS_SUCCESS) {std::cerr << "cublasCreate failed: " << stat << std::endl; gpu_destroy(ctx); return nullptr;}
  ctx->handle_created = true;

  cudaError_t err;
  //malloc for all device buffers
  err = cudaMalloc(&ctx->d_W1, in_dim*h1*sizeof(float));
  if (err != cudaSuccess) {std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl; gpu_destroy(ctx); return nullptr;}
  err = cudaMalloc(&ctx->d_W2, h1*h2*sizeof(float));
  if (err != cudaSuccess) {std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl; gpu_destroy(ctx); return nullptr;}
  err= cudaMalloc(&ctx->d_W3, h2*out_dim*sizeof(float));
  if (err != cudaSuccess) {std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl; gpu_destroy(ctx); return nullptr;}

  err = cudaMalloc(&ctx->d_b1, h1*sizeof(float));
  if (err != cudaSuccess) {std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl; gpu_destroy(ctx); return nullptr;}
  err = cudaMalloc(&ctx->d_b2, h2*sizeof(float));
  if (err != cudaSuccess) {std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl; gpu_destroy(ctx); return nullptr;}
  err = cudaMalloc(&ctx->d_b3, out_dim*sizeof(float));
  if (err != cudaSuccess) {std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl; gpu_destroy(ctx); return nullptr;}

  err = cudaMalloc(&ctx->d_X, maxB*in_dim*sizeof(float));
  if (err != cudaSuccess) {std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl; gpu_destroy(ctx); return nullptr;}
  err = cudaMalloc(&ctx->d_z1, maxB*h1*sizeof(float));
  if (err != cudaSuccess) {std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl; gpu_destroy(ctx); return nullptr;}
  err = cudaMalloc(&ctx->d_a1, maxB*h1*sizeof(float));
  if (err != cudaSuccess) {std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl; gpu_destroy(ctx); return nullptr;}
  err = cudaMalloc(&ctx->d_z2, maxB*h2*sizeof(float));
  if (err != cudaSuccess) {std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl; gpu_destroy(ctx); return nullptr;}
  err = cudaMalloc(&ctx->d_a2, maxB*h2*sizeof(float));
  if (err != cudaSuccess) {std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl; gpu_destroy(ctx); return nullptr;}
  err = cudaMalloc(&ctx->d_logits, maxB*out_dim*sizeof(float));
  if (err != cudaSuccess) {std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl; gpu_destroy(ctx); return nullptr;}

  return ctx;
}

