cmake_minimum_required(VERSION 3.6)
project(lm_framework)

# Alpha prototype executable
add_executable(lm_alpha
    repl.cpp
    config_io.cpp
)

target_include_directories(lm_alpha
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_link_libraries(lm_alpha
    PRIVATE
        lm_core
        nlohmann_json::nlohmann_json
)

# Copy config file to build directory
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/../../configs/alpha_config.json
    ${CMAKE_CURRENT_BINARY_DIR}/alpha_config.json
    COPYONLY
)

# Create run script for easy execution
file(GENERATE OUTPUT run_alpha.sh
    CONTENT "#!/bin/bash\ncd ${CMAKE_CURRENT_BINARY_DIR} && ./lm_alpha alpha_config.json"
    FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
)

