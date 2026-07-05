cmake_minimum_required(VERSION 3.16)
project(fmm_solver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

include_directories(${CMAKE_SOURCE_DIR}/include)

# --- Core library ---
add_library(fmm_core
    src/direct_sum.cpp
    src/octree.cpp
    src/spherical_harmonics.cpp
    src/multipole.cpp
    src/local.cpp
)

# --- Unit tests (Catch2, header-only, fetched at configure time) ---
option(FMM_BUILD_TESTS "Build C++ unit tests" ON)

if(FMM_BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.5.2
    )
    FetchContent_MakeAvailable(catch2)

    add_executable(fmm_tests
        tests/test_direct_sum.cpp
        tests/test_octree.cpp
        tests/test_spherical_harmonics.cpp
        tests/test_multipole.cpp
        tests/test_local.cpp
    )
    target_link_libraries(fmm_tests PRIVATE fmm_core Catch2::Catch2WithMain)
endif()

# --- Python bindings (pybind11), built only if requested ---
option(FMM_BUILD_PYTHON "Build pybind11 Python bindings" OFF)

if(FMM_BUILD_PYTHON)
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.12.0
    )
    FetchContent_MakeAvailable(pybind11)

    pybind11_add_module(fmm_py bindings/pybind_module.cpp)
    target_link_libraries(fmm_py PRIVATE fmm_core)
endif()