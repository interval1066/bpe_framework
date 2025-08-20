cmake_minimum_required(VERSION 3.5)
project(lm_framework)

# Add subdirectories for source components
add_subdirectory(tokenizer)
add_subdirectory(runtime)
add_subdirectory(alpha)

# Main library that combines all components
add_library(lm_core
    tokenizer/bpe_tokenizer.cpp
    tokenizer/unicode_utils.cpp
    runtime/init.cpp
    runtime/shutdown.cpp
)

target_include_directories(lm_core
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_link_libraries(lm_core
    PRIVATE
        nlohmann_json::nlohmann_json
        ICU::uc
        ICU::i18n
)

# Test executables
add_executable(test_bpe test_bpe.cpp)
target_link_libraries(test_bpe
    PRIVATE
        lm_core
        GTest::gtest_main
)

add_executable(test_unicode_bpe test_unicode_bpe.cpp)
target_link_libraries(test_unicode_bpe
    PRIVATE
        lm_core
        GTest::gtest_main
)

