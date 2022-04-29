cmake_minimum_required(VERSION 3.0.0)
project(run_demo VERSION 0.1.0)
# cross compiler for arm/aarch64 
#set(CMAKE_CXX_COMPILER "/opt/gcc-linaro-6.5.0-2018.12-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-g++")

enable_testing()
# onnxruntime dir
set(HOST_PACKAGE_DIR "/mnt/d/codes/c++/my_project/dpcrn_onnx_demo/onnxruntime-linux-x64-1.11.0")

include_directories(
        ${HOST_PACKAGE_DIR}/include/)

link_directories(
    ${HOST_PACKAGE_DIR}/lib/)

add_executable(run_demo run.cpp)

target_link_libraries(run_demo ${HOST_PACKAGE_DIR}/lib/libonnxruntime.so.1.11.0)


