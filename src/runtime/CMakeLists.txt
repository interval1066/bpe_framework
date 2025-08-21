project(lm_framework)
cmake_minimum_required(VERSION 3.6)

add_library(lm_runtime
    init.cpp
    shutdown.cpp
    state_utils.cpp  # Add this line
)

target_include_directories(lm_runtime
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_link_libraries(lm_runtime
    PRIVATE nlohmann_json::nlohmann_json
)


