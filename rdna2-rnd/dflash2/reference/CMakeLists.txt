cmake_minimum_required(VERSION 3.16)
project(dflash2_reference LANGUAGES C CXX ASM)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(BUILD_SHARED_LIBS OFF)
set(GGML_NATIVE ON)
set(GGML_OPENMP OFF)
set(GGML_BUILD_TESTS OFF)
set(GGML_BUILD_EXAMPLES OFF)
add_subdirectory(../../../third_party/llama.cpp/ggml ggml)

find_package(Threads REQUIRED)
add_library(ggml_qbridge SHARED ggml_qbridge.cpp)
target_compile_features(ggml_qbridge PRIVATE cxx_std_17)
target_include_directories(ggml_qbridge PRIVATE ../../../third_party/llama.cpp/ggml/include)
target_compile_options(ggml_qbridge PRIVATE -mavx2 -mfma -mf16c)
target_link_options(ggml_qbridge PRIVATE -Wl,--no-undefined)
target_link_libraries(ggml_qbridge PRIVATE ggml-cpu Threads::Threads)

option(DFLASH2_DONOR_PROBE "Compile the literal vendored graph probe" OFF)
if (DFLASH2_DONOR_PROBE)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    set(donor_tree "${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/llama.cpp")
    set(donor_inc "${CMAKE_CURRENT_BINARY_DIR}/literal_donor.inc")
    add_custom_command(OUTPUT "${donor_inc}"
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/extract_donor.py" "${donor_tree}" "${donor_inc}"
        DEPENDS extract_donor.py "${donor_tree}/src/models/dflash.cpp" "${donor_tree}/src/llama-graph.cpp"
                "${donor_tree}/src/llama-hparams.h"
        VERBATIM)
    add_library(ggml_dflash2_donor SHARED ggml_donor.cpp "${donor_inc}")
    target_compile_features(ggml_dflash2_donor PRIVATE cxx_std_17)
    target_include_directories(ggml_dflash2_donor PRIVATE "${donor_tree}/ggml/include" "${CMAKE_CURRENT_BINARY_DIR}")
    target_link_options(ggml_dflash2_donor PRIVATE -Wl,--no-undefined)
    target_link_libraries(ggml_dflash2_donor PRIVATE ggml-cpu Threads::Threads)
endif()
