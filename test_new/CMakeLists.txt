cmake_minimum_required(VERSION 3.0.0)
project(AKNN)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
add_definitions(-DTRAIN)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

link_directories(/usr/local/lib /usr/lib/x86_64-linux-gnu)

add_library(LIB aknn_test.cpp)
set_target_properties(LIB PROPERTIES PREFIX "")
set(SRCS test_new.cpp)

set(EXECUTABLE_OUTPUT_PATH ../)
set(TARGET_NAME "${PROJECT_NAME}.exe")
add_executable(${TARGET_NAME} ${SRCS})

target_link_libraries(${TARGET_NAME} LIB faiss.a
-Wl,--start-group libmkl_intel_lp64.a libmkl_gnu_thread.a libmkl_core.a
-Wl,--end-group -lgomp -lpthread -lm -ldl)
