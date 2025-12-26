#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "gpu_kernels.cu"

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
  bool params_uploaded;
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
    params_uploaded = false;

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

bool gpu_upload_params(GpuContext *ctx, const float *W1, const float *W2, const float *W3, 
    const float *b1, const float *b2, const float *b3) {

  if (ctx == nullptr) return false;
  cudaError_t err;

  err = cudaMemcpy(ctx->d_W1, W1, in_dim*h1*sizeof(float), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl; return false;}
  err = cudaMemcpy(ctx->d_W2, W2, h1*h2*sizeof(float), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl; return false;}
  err = cudaMemcpy(ctx->d_W3, W3, h2*out_dim*sizeof(float), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl; return false;}

  err = cudaMemcpy(ctx->d_b1, b1, h1*sizeof(float), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl; return false;}
  err = cudaMemcpy(ctx->d_b2, b2, h2*sizeof(float), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl; return false;}
  err = cudaMemcpy(ctx->d_b3, b3, out_dim*sizeof(float), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl; return false;}

  ctx->params_uploaded = true;

  return true;
}

void gpu_forward(GpuContext *ctx, const float *X_host, int B) {
  
  // safety check
  if (ctx == nullptr) return;
  if (B > ctx->maxB) {std::cout << "Batch Size too large for GPU" << std::endl; return;}
  if (!ctx->params_uploaded) {std::cout << " Params Not Yet Uploaded" << std::endl; return;}

  cudaError_t err;
  err = cudaMemcpy(ctx->d_X, X_host, B*in_dim*sizeof(float), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl; return;}

  int blockSize = 256;
  int numBlocks = (B*h1 + blockSize - 1)/blockSize;

  cublasStatus_t stat;
  const float alpha = 1.0f;
  const float beta = 0.0f;

  // layer 1 
  stat = cublasSgemm(ctx->handle, CUBLAS_OP_N, CUBLAS_OP_N, h1, B, in_dim, &alpha, ctx->d_W1, h1, ctx->d_X, in_dim, &beta, ctx->d_z1, h1);
  if (stat != CUBLAS_STATUS_SUCCESS) {std::cerr << "cublasSgemm failed: " << stat << std::endl; return;}
  biasAdd<<<numBlocks, blockSize>>>(ctx->d_z1, ctx->d_b1, B, h1);
  err = cudaGetLastError();
  if (err != cudaSuccess) {std::cerr << "cuda kernel biasAdd failed: " << cudaGetErrorString(err) << std::endl; return;}
  ReLU<<<numBlocks, blockSize>>>(ctx->d_a1, ctx->d_z1, B, h1);
  err = cudaGetLastError();
  if (err != cudaSuccess) {std::cerr << "cuda kernel ReLU failed: " << cudaGetErrorString(err) << std::endl; return;}
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {std::cerr << "cuda device Sync failed: " << cudaGetErrorString(err) << std::endl; return;}

  // layer 2 
  numBlocks = (B*h2 + blockSize - 1)/blockSize;
  stat = cublasSgemm(ctx->handle, CUBLAS_OP_N, CUBLAS_OP_N, h2, B, h1, &alpha, ctx->d_W2, h2, ctx->d_a1, h1, &beta, ctx->d_z2, h2);
  if (stat != CUBLAS_STATUS_SUCCESS) {std::cerr << "cublasSgemm failed: " << stat << std::endl; return;}
  biasAdd<<<numBlocks, blockSize>>>(ctx->d_z2, ctx->d_b2, B, h2);
  err = cudaGetLastError();
  if (err != cudaSuccess) {std::cerr << "cuda kernel biasAdd failed: " << cudaGetErrorString(err) << std::endl; return;}
  ReLU<<<numBlocks, blockSize>>>(ctx->d_a2, ctx->d_z2, B, h2);
  err = cudaGetLastError();
  if (err != cudaSuccess) {std::cerr << "cuda kernel ReLU failed: " << cudaGetErrorString(err) << std::endl; return;}
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {std::cerr << "cuda device Sync failed: " << cudaGetErrorString(err) << std::endl; return;}

  // layer 3 
  numBlocks = (B*out_dim + blockSize - 1)/blockSize;
  stat = cublasSgemm(ctx->handle, CUBLAS_OP_N, CUBLAS_OP_N, out_dim, B, h2, &alpha, ctx->d_W3, out_dim, ctx->d_a2, h2, &beta, ctx->d_logits, out_dim);
  if (stat != CUBLAS_STATUS_SUCCESS) {std::cerr << "cublasSgemm failed: " << stat << std::endl; return;}
  biasAdd<<<numBlocks, blockSize>>>(ctx->d_logits, ctx->d_b3, B, out_dim);
  err = cudaGetLastError();
  if (err != cudaSuccess) {std::cerr << "cuda kernel biasAdd failed: " << cudaGetErrorString(err) << std::endl; return;}
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {std::cerr << "cuda device Sync failed: " << cudaGetErrorString(err) << std::endl; return;}
}
