cmake_minimum_required(VERSION 3.18)

project(IMAGE_QUILTING LANGUAGES CXX CUDA)

add_subdirectory(lib/agz-utils)
target_compile_definitions(AGZUtils PUBLIC AGZ_UTILS_SSE)


file(GLOB_RECURSE SRC
    "${PROJECT_SOURCE_DIR}/src/*.cpp"
    "${PROJECT_SOURCE_DIR}/src/*.h"
    "${PROJECT_SOURCE_DIR}/src/*.cu"
)

add_executable(ImageQuilting ${SRC})

set_property(TARGET ImageQuilting PROPERTY CXX_STANDARD 17)
set_property(TARGET ImageQuilting PROPERTY CXX_STANDARD_REQUIRED ON)

target_include_directories(ImageQuilting PUBLIC "${PROJECT_SOURCE_DIR}/lib/cxxopts")
target_link_libraries(ImageQuilting PUBLIC AGZUtils)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(ImageQuilting PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(ImageQuilting PUBLIC IQ_USE_OPENMP=1)
endif()

include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_LIBRARIES "")
check_cxx_source_compiles("
  #include <filesystem>
  int main() { std::filesystem::path p{\".\"}; return 0; }
" HAS_STD_FILESYSTEM)
if(NOT HAS_STD_FILESYSTEM)
    target_link_libraries(ImageQuilting PUBLIC stdc++fs)
endif()

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set_target_properties(ImageQuilting PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION ON
    )
    target_compile_definitions(ImageQuilting PUBLIC IQ_USE_CUDA=1)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 75)
    endif()
else()
    target_compile_definitions(ImageQuilting PUBLIC IQ_USE_CUDA=0)
endif()

