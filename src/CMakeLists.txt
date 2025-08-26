cmake_minimum_required(VERSION 3.14)
project(lm_framework LANGUAGES CXX)

# Check for Intel x86-64 hardware
set(SUPPORTED_ARCHITECTURES x86_64 amd64 AMD64 i686 i386)
list(FIND SUPPORTED_ARCHITECTURES ${CMAKE_SYSTEM_PROCESSOR} ARCH_INDEX)
if(ARCH_INDEX EQUAL -1)
    message(FATAL_ERROR "This framework requires Intel x86-64 hardware. "
                        "Current processor architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

# Check for EIGEN_LOC variable
if(NOT DEFINED EIGEN_LOC)
    message(FATAL_ERROR "This framework requires the location of the Eigen header files. "
                        "Please set EIGEN_LOC to the path of your Eigen installation.")
elseif(EIGEN_LOC STREQUAL "")
    message(FATAL_ERROR "EIGEN_LOC is empty. Please set it to the path of your Eigen installation.")
endif()

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
    message(STATUS "Build type not specified, defaulting to Release")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable cross-directory linking
if(POLICY CMP0079)
    cmake_policy(SET CMP0079 NEW)
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${EIGEN_LOC} # Local Eigen installation
)

# Find dependencies
find_package(nlohmann_json 3.9 REQUIRED)
find_package(ICU REQUIRED COMPONENTS uc i18n)

# GoogleTest
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.11.0
)
FetchContent_MakeAvailable(googletest)

# Add subdirectories
add_subdirectory(src/tokenizer)
add_subdirectory(src/runtime)

# Header-only core components (Tensor implementation)
add_library(lm_core_components INTERFACE)
target_include_directories(lm_core_components INTERFACE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${EIGEN_LOC}  # Local Eigen installation
)

# Header-only model components
add_library(lm_model INTERFACE)
target_include_directories(lm_model INTERFACE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${EIGEN_LOC}  # Local Eigen installation
)
target_link_libraries(lm_model INTERFACE lm_core_components)

# Main library
add_library(lm_core
    src/runtime/init.cpp
    src/runtime/shutdown.cpp
    src/models/transformer.cpp          # Add Transformer implementation
    src/models/transformer_block.cpp    # Add Transformer block
    src/models/attention.cpp            # Add attention mechanism
    src/models/feed_forward.cpp         # Add feed forward network
    src/models/layer_norm.cpp           # Add layer normalization
)

target_link_libraries(lm_core
    PRIVATE
        lm_tokenizer
        lm_model
        lm_optimizers  # Add optimizers
        lm_models      # Add models
        lm_training    # Add training
        lm_integration_example
        nlohmann_json::nlohmann_json
)

# Set optimization flags for the core library
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(lm_core PRIVATE -O3)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_options(lm_core PRIVATE -DNDEBUG)
    endif()
endif()

# Test executables
add_executable(test_bpe src/test_bpe.cpp)
target_link_libraries(test_bpe
    PRIVATE
        lm_core
        GTest::gtest_main
)

add_executable(test_unicode_bpe src/test_unicode_bpe.cpp)
target_link_libraries(test_unicode_bpe
    PRIVATE
        lm_core
        GTest::gtest_main
)

# Alpha prototype executable
add_executable(lm_alpha
    src/alpha/repl.cpp
    src/alpha/config_io.cpp
)

target_link_libraries(lm_alpha
    PRIVATE
        lm_core
        nlohmann_json::nlohmann_json
)

# Install targets
install(TARGETS lm_core DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)

# Performance testing target
add_executable(performance_test src/performance_test.cpp)
target_link_libraries(performance_test
    PRIVATE
        lm_core
        GTest::gtest_main
)

# Integration example
add_executable(integration_example src/integration_example.cpp)
target_link_libraries(integration_example
    PRIVATE
        lm_core
)

# Add compiler warning flags
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Werror")
endif()

# Add coverage flags for debug builds
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    if(CMAKE_COMPILER_IS_GNUCXX)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-instr-generate -fcoverage-mapping")
    endif()
endif()

# Verify Eigen installation
add_custom_target(check_eigen
    COMMAND ${CMAKE_COMMAND} -E echo "Checking Eigen installation at ${EIGEN_LOC}"
    COMMAND test -f ${EIGEN_LOC}/Eigen/Core || (echo "Eigen not found at specified path: ${EIGEN_LOC}" && exit 1)
    COMMENT "Verifying Eigen installation"
)

# Make main targets depend on Eigen check
add_dependencies(lm_core check_eigen)
add_dependencies(test_bpe check_eigen)
add_dependencies(test_unicode_bpe check_eigen)
add_dependencies(lm_alpha check_eigen)
add_dependencies(performance_test check_eigen)
add_dependencies(integration_example check_eigen)
