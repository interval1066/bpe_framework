// src/tokenizer/unicode_utils.cpp
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

// Remove the "unicode::" qualification - we're already in the lm::unicode namespace
std::vector<std::string> unicode_split(const std::string& text) {
    std::vector<std::string> characters;
    int i = 0;
    while (i < text.length()) {
        int char_len = 1;
        // Check for UTF-8 multi-byte characters
        if ((text[i] & 0x80) == 0) {
            // ASCII character
            char_len = 1;
        } else if ((text[i] & 0xE0) == 0xC0) {
            // 2-byte UTF-8 character
            char_len = 2;
        } else if ((text[i] & 0xF0) == 0xE0) {
            // 3-byte UTF-8 character
            char_len = 3;
        } else if ((text[i] & 0xF8) == 0xF0) {
            // 4-byte UTF-8 character
            char_len = 4;
        }
        
        characters.push_back(text.substr(i, char_len));
        i += char_len;
    }
    return characters;
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

