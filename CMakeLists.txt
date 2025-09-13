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
include_directories(include/lm)
include_directories(include/lm/models)
include_directories(include/lm/training)
include_directories(include/lm/optimizers)
include_directories(include/lm/core)
include_directories(include/lm/tokenizer)
include_directories(include/lm/generation)
include_directories(include/lm/runtime)

# Find required packages
find_package(Eigen3 REQUIRED)
find_package(ICU REQUIRED COMPONENTS uc i18n)

# Cereal serialization library (header-only)
# We'll manually download it to avoid Boost dependency issues
if(NOT EXISTS ${CMAKE_SOURCE_DIR}/third_party/cereal/include/cereal/cereal.hpp)
    message(STATUS "Downloading Cereal library...")
    file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/third_party/cereal)
    
    # Download the specific version of Cereal
    file(DOWNLOAD 
        https://github.com/USCiLab/cereal/archive/refs/tags/v1.3.2.tar.gz
        ${CMAKE_SOURCE_DIR}/third_party/cereal_v1.3.2.tar.gz
        SHOW_PROGRESS
    )
    
    # Extract the archive
    execute_process(
        COMMAND tar -xf ${CMAKE_SOURCE_DIR}/third_party/cereal_v1.3.2.tar.gz -C ${CMAKE_SOURCE_DIR}/third_party
    )
    
    # Move the include directory
    file(RENAME 
        ${CMAKE_SOURCE_DIR}/third_party/cereal-1.3.2/include 
        ${CMAKE_SOURCE_DIR}/third_party/cereal/include
    )
    
    # Clean up
    file(REMOVE_RECURSE ${CMAKE_SOURCE_DIR}/third_party/cereal-1.3.2)
    file(REMOVE ${CMAKE_SOURCE_DIR}/third_party/cereal_v1.3.2.tar.gz)
endif()

# Add the manually downloaded Cereal include directory
set(CEREAL_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/third_party/cereal/include)
include_directories(${CEREAL_INCLUDE_DIR})
message(STATUS "Using Cereal from: ${CEREAL_INCLUDE_DIR}")

# Since Tensor is header-only, create an interface library for core components
add_library(lm_core INTERFACE)
target_include_directories(lm_core INTERFACE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(lm_core INTERFACE Eigen3::Eigen)

# Tokenizer library
add_library(lm_tokenizer STATIC
    src/tokenizer/bpe_tokenizer.cpp
    src/tokenizer/unicode_utils.cpp
)
target_link_libraries(lm_tokenizer PUBLIC lm_core ICU::uc ICU::i18n ${EIGEN3_LIBRARIES})

# Optimizers library
add_library(lm_optimizers STATIC
    src/optimizers/adam.cpp
)
target_link_libraries(lm_optimizers PUBLIC lm_core)

# Models library - keep only TransformerModel implementation
add_library(lm_models STATIC
    src/models/transformer_model.cpp
    src/models/conversation_model.cpp
)
target_link_libraries(lm_models PUBLIC lm_core lm_optimizers lm_tokenizer)

#add_library(lm_core INTERFACE)
#target_include_directories(lm_core INTERFACE ${CMAKE_SOURCE_DIR}/include)
#target_link_libraries(lm_core INTERFACE Eigen3::Eigen)

# Add TensorPool as part of the core library
target_sources(lm_core INTERFACE
    ${CMAKE_SOURCE_DIR}/include/lm/core/tensor_pool.hpp
)

# Generation library (samplers)
add_library(lm_generation STATIC
    src/generation/sampler.cpp
)
target_link_libraries(lm_generation PUBLIC lm_core)

# Context management library
add_library(lm_context STATIC
    src/context_manager.cpp
)
target_link_libraries(lm_context PUBLIC lm_core lm_tokenizer)

# Conversation management library
add_library(lm_conversation STATIC
    src/conversation_manager.cpp
)
target_link_libraries(lm_conversation PUBLIC lm_core lm_context)

# Runtime library
add_library(lm_runtime STATIC
    src/runtime/init.cpp
    src/runtime/shutdown.cpp
    src/runtime/state_utils.cpp
)
target_link_libraries(lm_runtime PUBLIC lm_core)

# Add Tensor and TensorPool as part of the core library
target_sources(lm_core INTERFACE
    ${CMAKE_SOURCE_DIR}/include/lm/core/tensor.hpp
    ${CMAKE_SOURCE_DIR}/include/lm/core/tensor_pool.hpp
)

# Alpha components
add_library(lm_alpha STATIC
    src/alpha/config_io.cpp
    src/alpha/repl.cpp
)
target_link_libraries(lm_alpha PUBLIC lm_core lm_runtime lm_conversation lm_models)

# Test executables
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
    lm_conversation
    lm_context
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

add_executable(sampler_test src/sampler_test.cpp)
target_link_libraries(sampler_test
    lm_training
    lm_models
    lm_optimizers
    lm_tokenizer
    lm_generation
    lm_core
)

add_executable(test_conversation src/test_conversation.cpp)
target_link_libraries(test_conversation
    lm_conversation
    lm_context
    lm_core
)

add_executable(test_logger src/test_logger.cpp)
target_link_libraries(test_logger
    lm_tokenizer
    lm_models
    lm_core
)

add_executable(test_transformer src/test_transformer.cpp)
target_link_libraries(test_transformer
    lm_models
    lm_tokenizer
    lm_core
)

add_executable(starter_convo src/starter_convo.cpp)
target_link_libraries(starter_convo
    lm_alpha
    lm_conversation
    lm_context
    lm_models
    lm_tokenizer
    lm_core
)

add_library(lm_training STATIC
    src/training/trainer.cpp
    src/training/data_loader.cpp
    src/training/losses.cpp
)

target_link_libraries(lm_training PUBLIC lm_models lm_optimizers lm_tokenizer)
add_executable(test_tensor_pool src/test_tensor_pool.cpp)
target_link_libraries(test_tensor_pool
    lm_core
)

# Enable testing if needed
#enable_testing()

# Print configuration summary
message(STATUS "Project configured successfully")
message(STATUS "Eigen3 found: ${Eigen3_FOUND}")
message(STATUS "ICU found: ${ICU_FOUND}")
message(STATUS "Cereal include: ${CEREAL_INCLUDE_DIR}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Profiling enabled: ${ENABLE_PROFILING}")

