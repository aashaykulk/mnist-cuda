
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

__global__ void reduce_sum(float *d_dlogits, float*d_db, int B, int n) {
  __shared__ float shmem[256];

  int tid = threadIdx.x;
  int j = blockIdx.x;
  if (j >= n) return;

  float local_sum = 0.0f;
  for (int i = tid; i < B; i+= blockDim.x) {
    local_sum += d_dlogits[i*n + j];
  }
  shmem[tid] = local_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
          shmem[tid] += shmem[tid + stride];
      }
      __syncthreads();
  }
  if (tid == 0) {
    d_db[j] = shmem[0];
  }
}

__global__ void ReLU_derivative(float *d_dz, float *d_da, float *d_a, int m, int n) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int total = m*n;
  if (t >= total) return;

  d_dz[t] =d_da[t]*(d_a[t] > 0);
}
