add_library(lm_runtime
    init.cpp
    shutdown.cpp
)

target_include_directories(lm_runtime
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_link_libraries(lm_runtime
    PRIVATE nlohmann_json::nlohmann_json
)
