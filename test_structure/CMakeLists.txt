cmake_minimum_required(VERSION 3.0.0)
project(AKNN)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

aux_source_directory(. SRCS)

set(EXECUTABLE_OUTPUT_PATH ../)
set(TARGET_NAME "${PROJECT_NAME}.exe")
add_executable(${TARGET_NAME} ${SRCS})

