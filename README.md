# mnist-cuda

A handwritten-digit classifier for **MNIST** implemented in **C++ with an optional CUDA backend**. The project includes:
- A minimal MNIST IDX loader (MSB-first / big-endian headers)
- CPU training + inference
- GPU training + inference via a CUDA backend
- Simple CLI binaries for training and testing

## Results (current checkpoints / runs)

These results are from the repo’s recorded runs:

| Backend | Task | Dataset slice | Accuracy | Wall time |
|---|---:|---:|---:|---:|
| CPU (M2 MacBook Air) | Train | MNIST train | 98.21% | 19m 27s |
| CPU (M2 MacBook Air) | Test | 256 test images | 97.06% | ~8.6s |
| GPU (NVIDIA RTX 5090) | Train | MNIST train | 97.18% | 1.663s |
| GPU (NVIDIA RTX 5090) | Test | 256 test images | 96.49% | (see screenshot) |

Screenshots:
- CPU train: `data/pics/train.png`
- CPU test: `data/pics/test.png`
- GPU train: `data/pics/train-gpu.png`
- GPU test: `data/pics/test-gpu.png`

> Notes:
> - “Test” is currently benchmarked on **256 images** (not the full 10,000-image MNIST test set). :contentReference[oaicite:0]{index=0}  
> - Timing shown above comes from the captured runs/shell `time` outputs. :contentReference[oaicite:1]{index=1}

## MNIST dataset

MNIST contains **60,000 training** and **10,000 test** 28×28 grayscale digit images (10 classes). :contentReference[oaicite:2]{index=2}  
The original IDX headers are stored **MSB-first (big-endian)**, so loaders must byte-swap on little-endian machines. :contentReference[oaicite:3]{index=3}

Expected files (either `.gz` or extracted):
- `train-images-idx3-ubyte(.gz)`
- `train-labels-idx1-ubyte(.gz)`
- `t10k-images-idx3-ubyte(.gz)`
- `t10k-labels-idx1-ubyte(.gz)` :contentReference[oaicite:4]{index=4}

## Build

### CPU build (macOS / Linux)
```bash
git clone https://github.com/aashaykulk/mnist-cuda
cd mnist-cuda
mkdir -p build && cd build
cmake ..
make -j
