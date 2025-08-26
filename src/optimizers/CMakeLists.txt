add_library(lm_optimizers
    adam.cpp
)

target_include_directories(lm_optimizers
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
        ${EIGEN_LOC}
)

target_link_libraries(lm_optimizers
    PUBLIC
        lm_core_components
)

