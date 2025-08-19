# Unicode Utilities Header File

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
/*```

This header provides a comprehensive set of Unicode utilities for the BPE tokenizer, including:

1. **Code point representation** with both numeric value and UTF-8 string
2. **Character classification** functions (whitespace, punctuation, control)
3. **Text normalization** using Unicode NFC form
4. **UTF-8 encoding/decoding** utilities
5. **Unicode-aware text splitting** for proper tokenization

These utilities enable the tokenizer to properly handle:
- Multi-byte UTF-8 sequences
- Unicode whitespace characters
- Text normalization for consistent processing
- Proper character boundary detection
- Support for various writing systems (Latin, CJK, Arabic, etc.)

The implementation in the corresponding `.cpp` file uses the ICU library for robust Unicode handling.*/
