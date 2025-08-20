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

