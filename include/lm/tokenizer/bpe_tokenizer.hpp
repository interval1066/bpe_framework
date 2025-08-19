# BPE Tokenizer Header File

Here's the complete `include/lm/tokenizer/bpe_tokenizer.hpp` file:

```cpp
#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <memory>
#include <utility>

namespace lm {

class BPETokenizer {
public:
    BPETokenizer();
    ~BPETokenizer();

    // Training methods
    void train(const std::vector<std::string>& corpus, size_t vocab_size = 30000);
    void train_from_file(const std::string& filename, size_t vocab_size = 30000);

    // Tokenization methods
    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& tokens) const;
    
    // Serialization
    bool save(const std::string& filename) const;
    bool load(const std::string& filename);

    // Vocabulary access
    size_t vocab_size() const;
    std::string id_to_token(int id) const;
    int token_to_id(const std::string& token) const;

    // Configuration
    void set_unknown_token(const std::string& token);
    void add_special_token(const std::string& token);
    
    // Unicode-specific methods
    void set_normalization(bool enabled);
    void set_byte_fallback(bool enabled);

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};

} // namespace lm
/*```

This header provides the complete interface for our BPE tokenizer with Unicode support. Key features include:

1. **Training methods** for building the vocabulary from text
2. **Encoding/decoding** for text-token conversion
3. **Serialization** for saving/loading trained models
4. **Vocabulary access** for inspecting tokens
5. **Configuration options** for special tokens and Unicode handling
6. **PIMPL pattern** for implementation hiding

The tokenizer supports:
- Unicode text processing with normalization
- Byte-level fallback for unknown characters
- Special token handling
- Custom vocabulary sizes
- Model persistence

This interface provides a clean abstraction while allowing for the complex Unicode handling implemented in the corresponding `.cpp` file. */