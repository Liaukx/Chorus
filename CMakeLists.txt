cmake_minimum_required(VERSION 3.5)

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
project(BLASTP LANGUAGES CXX C CUDA)

SET(CMAKE_BUILD_TYPE "Release")
option(USE_GPU_DIFFUSE "Use GPU Smith-waterman" ON)
if(USE_GPU_DIFFUSE)
    add_definitions(-DUSE_GPU_DIFFUSE)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -pthread")
if (NOT ("${CMAKE_SIZEOF_VOID_P}" STREQUAL "8"))
    message(SEND_ERROR "require 64 bit system")
endif()

if (CMAKE_BUILD_TYPE MATCHES Debug)
    message(STATUS "DEBUG MODE")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O0 -lineinfo -g -G  -Xptxas=-v")
else()
    message(STATUS "RELEASE MODE")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
endif()

find_package(OpenMP)
find_library(NVML_LIBRARY nvidia-ml)

message(STATUS "Use CPU local alignment")
set(SOURCES_UTIL
	    src/banded_smith_cpu.cpp
        src/util.cpp
        src/qit.cpp
        src/output.cpp
)

set(SOURCES_SEARCH
    src/hash_table.cu
    src/blastp.cu
    src/query_group.cu
)

set(SOURCES_CUDATOOL
    src/cudaTool.cu
)

add_library(util STATIC ${SOURCES_UTIL})
target_include_directories(util  PRIVATE 
        ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(util PRIVATE ${OpenMP_CXX_LIBRARIES})
target_include_directories (util
	PRIVATE
		 ${OpenMP_CXX_INCLUDE_DIRS}
)
target_compile_options (util
	PRIVATE
		${OpenMP_CXX_FLAGS}
)

add_executable(query src/main.cpp ${SOURCES_SEARCH})
target_link_libraries(query util)
target_link_libraries(query ${NVML_LIBRARY})

add_executable(createDB src/createDB.cpp ${SOURCES_CUDATOOL})

target_include_directories(query
    PRIVATE 
        ${PROJECT_SOURCE_DIR}/include
)

target_include_directories(createDB
    PRIVATE 
        ${PROJECT_SOURCE_DIR}/include
)

