add_library(lm_training
    trainer.cpp
)

target_include_directories(lm_training
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
        ${EIGEN_LOC}
)

target_link_libraries(lm_training
    PUBLIC
        lm_core_components
        lm_models
        lm_optimizers
)

