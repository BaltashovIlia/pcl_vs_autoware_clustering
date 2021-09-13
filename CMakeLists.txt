cmake_minimum_required(VERSION 3.10)
project(pcl_vs_autoware)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

find_package(PCL REQUIRED)
find_package(benchmark REQUIRED)
find_package(CUDA REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

cuda_add_library(gpu_euclidean_clustering src/gpu_euclidean_clustering.h src/gpu_euclidean_clustering.cu)

add_executable(pcl_vs_autoware_bench src/bench_main.cpp)
target_link_libraries(pcl_vs_autoware_bench ${PCL_LIBRARIES} benchmark::benchmark gpu_euclidean_clustering)

add_executable(pcl_vs_autoware_compare src/compare_main.cpp)
target_link_libraries(pcl_vs_autoware_compare ${PCL_LIBRARIES} benchmark::benchmark gpu_euclidean_clustering)