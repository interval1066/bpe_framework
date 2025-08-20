# Tokenizer library
add_library(lm_tokenizer
    bpe_tokenizer.cpp
    unicode_utils.cpp
)

target_include_directories(lm_tokenizer
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_link_libraries(lm_tokenizer
    PRIVATE 
        ICU::uc 
        ICU::i18n
)

# CPU-specific optimization flags
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # Enable aggressive optimizations
    target_compile_options(lm_tokenizer PRIVATE -O3 -march=native)
    
    # Enable SSE4.2 instructions if available
    target_compile_options(lm_tokenizer PRIVATE -msse4.2)
    
    # Enable link-time optimization
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.9)
        target_compile_options(lm_tokenizer PRIVATE -flto)
        set_target_properties(lm_tokenizer PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif()
    
    # Enable specific optimizations for GCC
    if(CMAKE_COMPILER_IS_GNUCXX)
        target_compile_options(lm_tokenizer PRIVATE -ftree-vectorize -funroll-loops)
    endif()
    
    # Enable specific optimizations for Clang
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(lm_tokenizer PRIVATE -Rpass=.* -Rpass-missed=.* -Rpass-analysis=.*)
    endif()
endif()

# Add profiling support
if(PROFILE)
    if(CMAKE_COMPILER_IS_GNUCXX)
        target_compile_options(lm_tokenizer PRIVATE -pg)
        target_link_options(lm_tokenizer PRIVATE -pg)
    endif()
endif()
