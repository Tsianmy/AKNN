cmake_minimum_required(VERSION 3.0.0)
project(AKNN)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

aux_source_directory(. SRCS)

set(EXECUTABLE_OUTPUT_PATH ../)
set(TARGET_NAME "${PROJECT_NAME}.exe")
add_executable(${TARGET_NAME} ${SRCS})

