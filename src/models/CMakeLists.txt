add_library(lm_models
    transformer.cpp
    transformer_block.cpp
    attention.cpp
    feed_forward.cpp
    layer_norm.cpp
)

target_include_directories(lm_models
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
        ${EIGEN_LOC}
)

target_link_libraries(lm_models
    PUBLIC
        lm_core_components
)

