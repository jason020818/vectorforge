cmake_minimum_required(VERSION 3.24)

project(vectorforge
    VERSION 0.1.0
    DESCRIPTION "Open-source ANN retrieval engine with a public performance competition framework"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(VECTORFORGE_BUILD_TESTS "Build C++ unit tests" ON)
option(VECTORFORGE_BUILD_PYTHON "Build Python extension module" OFF)
option(VECTORFORGE_ENABLE_SANITIZERS "Enable ASan/UBSan (Debug-oriented)" OFF)

include(GNUInstallDirs)

add_library(vectorforge
    src/distance/scalar.cpp
    src/index/flat_index.cpp
    src/distance/avx2.cpp
    src/distance/avx512.cpp
)

add_library(vectorforge::vectorforge ALIAS vectorforge)

target_include_directories(vectorforge
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_features(vectorforge PUBLIC cxx_std_20)

if(MSVC)
    target_compile_options(vectorforge PRIVATE /W4 /permissive-)
else()
    target_compile_options(vectorforge PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

if(VECTORFORGE_ENABLE_SANITIZERS AND NOT MSVC)
    target_compile_options(vectorforge PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(vectorforge PUBLIC -fsanitize=address,undefined)
endif()

if(VECTORFORGE_BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.8.1
    )
    FetchContent_MakeAvailable(Catch2)
    list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras")
    enable_testing()
    add_subdirectory(tests)
endif()

if(VECTORFORGE_BUILD_PYTHON)
    set(PYBIND11_FINDPYTHON ON)
    find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(_vectorforge python/bindings.cpp)
    target_link_libraries(_vectorforge PRIVATE vectorforge)
    target_compile_definitions(_vectorforge PRIVATE
        VECTORFORGE_VERSION="${PROJECT_VERSION}"
    )
    install(TARGETS _vectorforge DESTINATION vectorforge)
    install(FILES python/vectorforge/__init__.py DESTINATION vectorforge)
endif()
