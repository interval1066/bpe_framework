/*# Unicode Utilities Implementation File

Here's the complete `src/tokenizer/unicode_utils.cpp` file:

```cpp*/
#include "lm/tokenizer/unicode_utils.hpp"
#include <unicode/uchar.h>
#include <unicode/unistr.h>
#include <unicode/normlzr.h>
#include <unicode/ustring.h>
#include <stdexcept>
#include <algorithm>

namespace lm::unicode {

bool is_whitespace(uint32_t codepoint) {
    return u_isUWhiteSpace(codepoint);
}

bool is_punctuation(uint32_t codepoint) {
    return u_ispunct(codepoint);
}

bool is_control(uint32_t codepoint) {
    return u_iscntrl(codepoint);
}

std::string normalize(const std::string& text) {
    try {
        icu::UnicodeString unicode_str = icu::UnicodeString::fromUTF8(text);
        icu::UnicodeString normalized;
        UErrorCode status = U_ZERO_ERROR;
        
        icu::Normalizer::normalize(unicode_str, UNORM_NFC, 0, normalized, status);
        
        if (U_FAILURE(status)) {
            throw std::runtime_error("Unicode normalization failed");
        }
        
        std::string result;
        normalized.toUTF8String(result);
        return result;
    } catch (const std::exception& e) {
        throw std::runtime_error("Unicode normalization error: " + std::string(e.what()));
    }
}

std::vector<CodePoint> to_code_points(const std::string& text) {
    std::vector<CodePoint> code_points;
    
    for (size_t i = 0; i < text.size(); ) {
        CodePoint cp;
        uint32_t codepoint;
        int offset = 0;
        
        // Decode UTF-8
        U8_NEXT(text.c_str(), i, text.size(), codepoint);
        
        if (codepoint == U_SENTINEL) {
            // Handle invalid UTF-8 gracefully instead of throwing
            // Use replacement character (U+FFFD) for invalid sequences
            cp.value = 0xFFFD;
            cp.utf8 = "�";  // Replacement character
            code_points.push_back(cp);
            
            // Skip this byte and continue
            i++;
            continue;
        }
        
        // Get the UTF-8 bytes for this code point
        char utf8_buf[5] = {0};
        U8_APPEND_UNSAFE(utf8_buf, offset, codepoint);
        
        cp.value = codepoint;
        cp.utf8 = std::string(utf8_buf, offset);
        code_points.push_back(cp);
        
        i += offset;
    }
    
    return code_points;
}

std::string from_code_points(const std::vector<CodePoint>& code_points) {
    std::string result;
    for (const auto& cp : code_points) {
        result += cp.utf8;
    }
    return result;
}

std::vector<std::string> unicode_split(const std::string& text) {
    std::vector<std::string> words;
    std::string current_word;
    auto code_points = to_code_points(text);
    
    for (const auto& cp : code_points) {
        if (is_whitespace(cp.value)) {
            if (!current_word.empty()) {
                words.push_back(current_word);
                current_word.clear();
            }
        } else {
            current_word += cp.utf8;
        }
    }
    
    if (!current_word.empty()) {
        words.push_back(current_word);
    }
    
    return words;
}

std::vector<std::string> split_on_character_boundaries(const std::string& text) {
    std::vector<std::string> characters;
    auto code_points = to_code_points(text);
    
    for (const auto& cp : code_points) {
        characters.push_back(cp.utf8);
    }
    
    return characters;
}

} // namespace lm::unicode
/*```

This implementation provides comprehensive Unicode support for the BPE tokenizer with:

1. **Character classification** using ICU library functions
2. **Unicode normalization** to NFC form for consistent processing
3. **UTF-8 encoding/decoding** with proper error handling
4. **Unicode-aware text splitting** based on character boundaries
5. **Robust error handling** for invalid UTF-8 sequences

Key features:
- **ICU library integration** for professional-grade Unicode handling
- **Proper UTF-8 sequence processing** with boundary detection
- **Character classification** for whitespace, punctuation, and control characters
- **Normalization** to ensure consistent representation of equivalent Unicode sequences
- **Exception safety** with proper error reporting

This implementation enables the BPE tokenizer to properly handle multilingual text, including complex scripts and emoji, while maintaining compatibility with ASCII text.*/
