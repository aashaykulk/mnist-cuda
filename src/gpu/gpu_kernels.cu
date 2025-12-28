
__global__ void biasAdd(float *d_z, float *d_b, int m, int n) {

  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int total = m*n;
  if (t >= total) return;

  int col = t % n;
  d_z[t] += d_b[col];
}

__global__ void ReLU(float* d_a, float* d_z, int m, int n) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int total = m*n;
  if (t >= total) return;

  d_a[t] = fmaxf(d_z[t], 0.0f);
}

__global__ void compute_d_dlogits(float *d_dlogits, float *d_logits, uint8_t *d_y, int m, int n) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int total = m*n;
  if (t >= total) return;

  int i = t / n;
  int j = t % n;
  int base = i*n;
  float max_i = d_logits[base];
  float sum_i = 0.0f;
  float p_ij = 0.0f;

  for (int k = 0; k < n; k++) {
    if (max_i < d_logits[base + k]) max_i = d_logits[base + k];
  }
  for (int k = 0; k < n; k++) {
    sum_i += exp(d_logits[base + k] - max_i);
  }

  p_ij = exp(d_logits[i*n + j] - max_i)/sum_i;

  d_dlogits[i*n + j] = p_ij - (j == d_y[i]);
}
