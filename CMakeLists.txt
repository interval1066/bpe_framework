cmake_minimum_required(VERSION 3.14)
project(lm_framework LANGUAGES CXX)

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

# Main library
add_library(lm_core
    src/runtime/init.cpp
    src/runtime/shutdown.cpp
)

target_link_libraries(lm_core
    PRIVATE
        lm_tokenizer
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
