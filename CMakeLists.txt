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
find_package(Threads REQUIRED)

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

# Configuration management library
add_library(lm_config STATIC
    src/config_manager.cpp
)
target_link_libraries(lm_config PUBLIC lm_core)

# Tokenizer library
add_library(lm_tokenizer STATIC
    src/tokenizer/bpe_tokenizer.cpp
    src/tokenizer/unicode_utils.cpp
)
target_link_libraries(lm_tokenizer PUBLIC lm_core lm_config ICU::uc ICU::i18n ${EIGEN3_LIBRARIES})

# Optimizers library
add_library(lm_optimizers STATIC
    src/optimizers/adam.cpp
)
target_link_libraries(lm_optimizers PUBLIC lm_core)

add_library(lm_models STATIC
    src/models/transformer_model.cpp
    src/models/conversation_model.cpp
    src/context/context_manager.cpp  # Add this line
)
target_link_libraries(lm_models PUBLIC lm_core lm_optimizers lm_tokenizer)

# Add TensorPool as part of the core library
target_sources(lm_core INTERFACE
    ${CMAKE_SOURCE_DIR}/include/lm/core/tensor_pool.hpp
)

# Generation library (samplers)
add_library(lm_generation STATIC
    src/generation/sampler.cpp
    src/generation/greedy_sampler.cpp
    src/generation/temperature_sampler.cpp
    src/generation/topk_sampler.cpp
    src/generation/topp_sampler.cpp
    src/generation/random_sampler.cpp  # Add this line
)
target_link_libraries(lm_generation PUBLIC lm_core)

# Add header files to the generation library
target_sources(lm_generation INTERFACE
    ${CMAKE_SOURCE_DIR}/include/lm/generation/sampler.hpp
    ${CMAKE_SOURCE_DIR}/include/lm/generation/greedy_sampler.hpp
    ${CMAKE_SOURCE_DIR}/include/lm/generation/temperature_sampler.hpp
    ${CMAKE_SOURCE_DIR}/include/lm/generation/topk_sampler.hpp
    ${CMAKE_SOURCE_DIR}/include/lm/generation/topp_sampler.hpp
    ${CMAKE_SOURCE_DIR}/include/lm/generation/random_sampler.hpp  # Add this line
)

# Context management library
add_library(lm_context STATIC
    src/context/context_manager.cpp
)
target_link_libraries(lm_context PUBLIC lm_core lm_tokenizer)

# Conversation management library
add_library(lm_conversation STATIC
    src/conversation/conversation_manager.cpp
)
target_link_libraries(lm_conversation PUBLIC lm_core lm_context)

add_library(lm_inference STATIC
    src/inference/inference_engine.cpp
)
target_link_libraries(lm_inference PUBLIC lm_conversation lm_models lm_tokenizer lm_generation)

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
target_link_libraries(lm_alpha PUBLIC lm_core lm_runtime lm_conversation lm_models lm_config)

# Training library
add_library(lm_training STATIC
    src/training/trainer.cpp
    src/training/data_loader.cpp
    src/training/losses.cpp
)
target_link_libraries(lm_training PUBLIC lm_models lm_optimizers lm_tokenizer lm_config)

# Test executables
add_executable(performance_test tests/performance_test.cpp)
target_link_libraries(performance_test
    lm_training
    lm_models
    lm_optimizers
    lm_tokenizer
    lm_core
)

add_executable(test_data_loader tests/test_data_loader.cpp)
target_link_libraries(test_data_loader
    lm_training
    lm_tokenizer
    lm_core
)

add_executable(chat_demo tests/chat_demo.cpp)
target_link_libraries(chat_demo
    lm_inference
    lm_conversation
    lm_models
    lm_tokenizer
    lm_generation
    lm_core
)

add_executable(test_generation tests/test_generation.cpp)
target_link_libraries(test_generation
    lm_training
    lm_models
    lm_optimizers
    lm_tokenizer
    lm_generation
    lm_core
)

add_executable(serialization_demo tests/serialization_demo.cpp)
target_link_libraries(serialization_demo
    lm_training
    lm_models
    lm_optimizers
    lm_tokenizer
    lm_conversation
    lm_context
    lm_core
    lm_generation  # Add this line
)

# Update other executables that might use samplers
target_link_libraries(chat_demo
    lm_inference
    lm_conversation
    lm_models
    lm_tokenizer
    lm_generation  # Ensure this is included
    lm_core
)

target_link_libraries(test_generation
    lm_training
    lm_models
    lm_optimizers
    lm_tokenizer
    lm_generation
    lm_core
)

add_executable(test_bpe tests/test_bpe.cpp)
target_link_libraries(test_bpe
    lm_tokenizer
    lm_core
)

add_executable(test_unicode_bpe tests/test_unicode_bpe.cpp)
target_link_libraries(test_unicode_bpe
    lm_tokenizer
    lm_core
)

add_executable(sampler_test tests/sampler_test.cpp)
target_link_libraries(sampler_test
    lm_training
    lm_models
    lm_optimizers
    lm_tokenizer
    lm_generation
    lm_core
)

add_executable(test_conversation tests/test_conversation.cpp)
target_link_libraries(test_conversation
    lm_conversation
    lm_models
    lm_tokenizer
    lm_generation
    lm_core
)

add_executable(test_logger tests/test_logger.cpp)
target_link_libraries(test_logger
    lm_tokenizer
    lm_models
    lm_core
)

add_executable(test_transformer tests/test_transformer.cpp)
target_link_libraries(test_transformer
    lm_models
    lm_tokenizer
    lm_core
)

add_executable(starter_convo tests/starter_convo.cpp)
target_link_libraries(starter_convo
    lm_alpha
    lm_conversation
    lm_context
    lm_models
    lm_tokenizer
    lm_core
)

# Add the interactive learning conversation application
add_executable(interactive_learning_convo tests/interactive_learning_convo.cpp)
target_link_libraries(interactive_learning_convo
    lm_alpha
    lm_conversation
    lm_context
    lm_models
    lm_tokenizer
    lm_training
    lm_generation
    lm_config  # Add config manager
    lm_core
    Threads::Threads  # Add threading support
)

add_executable(test_tensor_pool tests/test_tensor_pool.cpp)
target_link_libraries(test_tensor_pool
    lm_core
)

# Add test for config manager
add_executable(test_config_manager tests/test_config_manager.cpp)
target_link_libraries(test_config_manager
    lm_config
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

