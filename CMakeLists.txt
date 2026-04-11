cmake_minimum_required(VERSION 3.18...4.0)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(Python COMPONENTS NumPy REQUIRED)

pybind11_add_module(_core MODULE
    src/libcubp/main.cpp
    src/libcubp/kernels.cu
)

target_include_directories(_core PRIVATE
    ${Python_NumPy_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/libcubp
)

target_link_libraries(_core PRIVATE CUDA::cudart)

install(TARGETS _core DESTINATION ${SKBUILD_PROJECT_NAME})
