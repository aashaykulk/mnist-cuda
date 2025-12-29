#pragma once

struct GpuContext;

//creates pre-allocated buffers for the biggest batch
GpuContext *gpu_create(int maxB);
void gpu_destroy(GpuContext *ctx);

bool gpu_is_available();

// ML ALgos
bool gpu_upload_params(GpuContext *ctx, const float *W1, const float *W2, const float *W3, const float *b1, const float *b2, const float *b3);

void gpu_forward(GpuContext *ctx, const float *X_host, int B);
void gpu_download_logits(GpuContext *ctx, float *logits_host, int B);
void gpu_compute_d_dlogits(GpuContext *ctx, int B); 
void gpu_backward(GpuContext *ctx, const float *dlogits_host, int B);
void gpu_step(GpuContext *ctx, const float lr, int B);
