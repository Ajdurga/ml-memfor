cmake_minimum_required(VERSION 3.18)
project(cifar_infer CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "-g O0")
#Path to your LibTorch folder (the one that contains share/cmake/Torch)
# Do NOT change this unless your libtorch lives somewhere else
set(Torch_DIR "${CMAKE_SOURCE_DIR}/../libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

add_executable(cifar_infer src/main.cpp)
target_link_libraries(cifar_infer PRIVATE Torch::Torch)
