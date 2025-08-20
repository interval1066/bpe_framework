cmake_minimum_required(VERSION 3.5)
project(lm_framework)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Find dependencies
find_package(nlohmann_json 3.9 REQUIRED)

# Find ICU library
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
    src/tokenizer/bpe_tokenizer.cpp
    src/tokenizer/unicode_utils.cpp
    src/runtime/init.cpp
    src/runtime/shutdown.cpp
)

target_link_libraries(lm_core
    PRIVATE
        nlohmann_json::nlohmann_json
        ICU::uc
        ICU::i18n
)

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

