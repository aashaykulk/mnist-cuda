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

