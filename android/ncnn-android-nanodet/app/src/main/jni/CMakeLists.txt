project(gpt2chat)

cmake_minimum_required(VERSION 3.10)

set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-20220216-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

add_library(gpt2chat SHARED gpt2chat.cpp gpt2.cpp)

target_link_libraries(gpt2chat ncnn)
