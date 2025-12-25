cmake_minimum_required(VERSION 3.16)
project(mnist_nn LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- nn library ---
add_library(nn STATIC
  src/neural_network/network.cpp
)
target_include_directories(nn PUBLIC src)

# --- mnist loader library ---
add_library(mnist STATIC
  src/mnist/mnist_loader.cpp
)
target_include_directories(mnist PUBLIC src)

# --- executable: training ---
add_executable(train
  src/train.cpp
)

# --- executable: testing ---
add_executable(test
  src/test.cpp
)
target_link_libraries(train PRIVATE nn mnist)
target_link_libraries(test PRIVATE nn mnist)
