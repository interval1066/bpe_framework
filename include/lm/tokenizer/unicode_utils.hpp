//# Unicode Utilities Header File

#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace lm::unicode {

// Unicode character representation
struct CodePoint {
    uint32_t value;
    std::string utf8;  // UTF-8 representation
};

// Check if a code point is whitespace
bool is_whitespace(uint32_t codepoint);

// Check if a code point is punctuation
bool is_punctuation(uint32_t codepoint);

// Check if a code point is a control character
bool is_control(uint32_t codepoint);

// Normalize Unicode text (NFC normalization)
std::string normalize(const std::string& text);

// Split text into Unicode code points
std::vector<CodePoint> to_code_points(const std::string& text);

// Convert code points back to UTF-8 string
std::string from_code_points(const std::vector<CodePoint>& code_points);

// Unicode-aware string split (handles Unicode whitespace)
std::vector<std::string> unicode_split(const std::string& text);

// Unicode-aware character boundaries
std::vector<std::string> split_on_character_boundaries(const std::string& text);

} // namespace lm::unicode

