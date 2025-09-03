cmake_minimum_required(VERSION 3.16)
project(bpe_framework)

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "AMD64")
    add_compile_definitions(__x86_64__)
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add profile build option - must be defined before any usage
option(ENABLE_PROFILING "Enable profiling with gprof" OFF)

# Set compiler flags based on build type and profiling option
if(ENABLE_PROFILING)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")
    message(STATUS "Profiling enabled: gprof flags added")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG")
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -g")
elseif(CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Os -DNDEBUG")
endif()

# Include directories
include_directories(include)

# Find required packages
find_package(Eigen3 REQUIRED)
find_package(ICU REQUIRED COMPONENTS uc i18n)

# Since Tensor is header-only, create an interface library for core components
add_library(lm_core INTERFACE)
target_include_directories(lm_core INTERFACE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(lm_core INTERFACE Eigen3::Eigen)

# Tokenizer library
add_library(lm_tokenizer STATIC
    src/tokenizer/bpe_tokenizer.cpp
    src/tokenizer/unicode_utils.cpp
)
target_link_libraries(lm_tokenizer PUBLIC lm_core ICU::uc ICU::i18n)

# Optimizers library
add_library(lm_optimizers STATIC
    src/optimizers/adam.cpp
)
target_link_libraries(lm_optimizers PUBLIC lm_core)

# Models library - add Transformer implementations
add_library(lm_models STATIC
    src/models/language_model.cpp
    src/models/transformer.cpp          # Add this if it exists
    src/models/transformer_block.cpp    # Add this if it exists
)
target_link_libraries(lm_models PUBLIC lm_core lm_optimizers)

# Generation library (samplers)
add_library(lm_generation STATIC
    src/generation/sampler.cpp
)
target_link_libraries(lm_generation PUBLIC lm_core)

# Training library
add_library(lm_training STATIC
    src/training/trainer.cpp
)
target_link_libraries(lm_training PUBLIC lm_models lm_tokenizer lm_optimizers lm_generation)

add_executable(performance_test src/performance_test.cpp)
target_link_libraries(performance_test
    lm_training
    lm_models
    lm_optimizers
    lm_tokenizer
    lm_core
)

add_executable(test_generation src/test_generation.cpp)
target_link_libraries(test_generation
    lm_training
    lm_models
    lm_optimizers
    lm_tokenizer
    lm_generation
    lm_core
)

add_executable(serialization_demo src/serialization_demo.cpp)
target_link_libraries(serialization_demo
    lm_training
    lm_models
    lm_optimizers
    lm_tokenizer
    lm_core
)

add_executable(test_bpe src/test_bpe.cpp)
target_link_libraries(test_bpe
    lm_tokenizer
    lm_core
)

add_executable(test_unicode_bpe src/test_unicode_bpe.cpp)
target_link_libraries(test_unicode_bpe
    lm_tokenizer
    lm_core
)

# Enable testing if needed
enable_testing()

# Print configuration summary
message(STATUS "Project configured successfully")
message(STATUS "Eigen3 found: ${Eigen3_FOUND}")
message(STATUS "ICU found: ${ICU_FOUND}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Profiling enabled: ${ENABLE_PROFILING}")

