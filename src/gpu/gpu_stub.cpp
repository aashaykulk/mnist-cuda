// src/gpu/gpu_stub.cpp
#include <cstdio>
#include <cstdlib>

// Forward-declare the type exactly as in your headers.
struct GpuContext;

static void no_cuda(const char* fn) {
  std::fprintf(stderr, "[GPU stub] %s called, but USE_CUDA=OFF (no CUDA backend linked).\n", fn);
  std::fflush(stderr);
}

GpuContext* gpu_create(int /*maxB*/) {
  no_cuda("gpu_create");
  return nullptr;
}

void gpu_destroy(GpuContext* /*ctx*/) {
  no_cuda("gpu_destroy");
}

void gpu_upload_params(GpuContext* /*ctx*/,
                       const float* /*W1*/, const float* /*b1*/,
                       const float* /*W2*/, const float* /*b2*/,
                       const float* /*W3*/, const float* /*b3*/) {
  no_cuda("gpu_upload_params");
}

void gpu_download_params(GpuContext* /*ctx*/,
                         float* /*W1*/, float* /*b1*/,
                         float* /*W2*/, float* /*b2*/,
                         float* /*W3*/, float* /*b3*/) {
  no_cuda("gpu_download_params");
}

void gpu_forward(GpuContext* /*ctx*/, const float* /*X*/, int /*B*/) {
  no_cuda("gpu_forward");
}

void gpu_backward(GpuContext* /*ctx*/, const unsigned char* /*y*/, int /*B*/) {
  no_cuda("gpu_backward");
}

void gpu_download_logits(GpuContext* /*ctx*/, float* /*logits*/, int /*B*/) {
  no_cuda("gpu_download_logits");
}

void gpu_step(GpuContext* /*ctx*/, float /*lr*/, int /*B*/) {
  no_cuda("gpu_step");
}
