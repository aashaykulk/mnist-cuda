cmake_minimum_required(VERSION 3.16)
project(mnist_nn LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_CUDA "Enable CUDA backend" OFF)

# ---------------- CPU NN library ----------------
add_library(nn_cpu STATIC
  src/neural_network/network.cpp
)
target_include_directories(nn_cpu PUBLIC src)

# ---------------- MNIST loader ------------------
add_library(mnist STATIC
  src/mnist/mnist_loader.cpp
)
target_include_directories(mnist PUBLIC src)

# ---------------- CUDA backend (optional) -------
if (USE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

  add_library(nn_cuda STATIC
    src/gpu/gpu_ops.cu
  )
  target_include_directories(nn_cuda PUBLIC src)
  set_target_properties(nn_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(nn_cuda PUBLIC CUDA::cudart CUDA::cublas)

  # If you use #ifdef USE_CUDA in C++ sources, define it there too
  target_compile_definitions(nn_cpu PUBLIC USE_CUDA)
else()
  add_library(nn_cuda_stub STATIC
    src/gpu/gpu_stub.cpp
  )
  target_include_directories(nn_cuda_stub PUBLIC src)
endif()

# ---------------- Train executable --------------
add_executable(train src/train.cpp)

if (USE_CUDA)
  target_link_libraries(train PRIVATE nn_cpu nn_cuda mnist)
else()
  target_link_libraries(train PRIVATE nn_cpu nn_cuda_stub mnist)
endif()
# ---------------- Test executable ---------------
add_executable(test src/test.cpp)

if (USE_CUDA)
  target_compile_definitions(train PRIVATE USE_CUDA)
  target_compile_definitions(test  PRIVATE USE_CUDA)

  target_link_libraries(train PRIVATE nn_cpu nn_cuda mnist)
  target_link_libraries(test  PRIVATE nn_cpu nn_cuda mnist)
else()
  target_link_libraries(train PRIVATE nn_cpu nn_cuda_stub mnist)
  target_link_libraries(test  PRIVATE nn_cpu nn_cuda_stub mnist)
endif()
