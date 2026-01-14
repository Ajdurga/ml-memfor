cmake_minimum_required(VERSION 3.18)
project(cifar_infer CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
 
# Point to libtorch (folder that contains share/cmake/Torch/TorchConfig.cmake)
set(Torch_DIR "${CMAKE_SOURCE_DIR}/../libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
 
add_executable(cifar_infer src/main.cpp)
 
# ---- Robust linking style (works across LibTorch variants) ----
target_link_libraries(cifar_infer PRIVATE "${TORCH_LIBRARIES}")
# ---------------------------------------------------------------
 
# (Optional) some packages advertise extra flags
if (TORCH_CXX_FLAGS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
endif()