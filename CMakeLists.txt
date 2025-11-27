cmake_minimum_required(VERSION 3.16)
project(mnist_nn LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# nn library
add_library(nn STATIC
    src/nn/network.cpp
)
target_include_directories(nn PUBLIC src)

# mnist loader library
add_library(mnist STATIC
    src/mnist/mnist_loader.cpp
)
target_include_directories(mnist PUBLIC src)

# CPU executable
add_executable(mnist_cpu
    src/main_cpu.cpp
)
target_link_libraries(mnist_cpu PRIVATE nn mnist)
