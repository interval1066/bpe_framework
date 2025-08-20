#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <memory>
#include <utility>
#include <cstdint>  // For uint16_t

namespace lm {
    
using TokenID = uint16_t;  // Support for 65k vocabulary

class BPETokenizer {
public:
    BPETokenizer();
    ~BPETokenizer();

    // Training methods
    void train(const std::vector<std::string>& corpus, size_t vocab_size = 30000);
    void train_from_file(const std::string& filename, size_t vocab_size = 30000);

    // Tokenization methods
    std::vector<TokenID> encode(const std::string& text) const;
    std::string decode(const std::vector<TokenID>& tokens) const;
    
    // Serialization
    bool save(const std::string& filename) const;
    bool load(const std::string& filename);

    // Vocabulary access
    size_t vocab_size() const;
    std::string id_to_token(TokenID id) const;
    TokenID token_to_id(const std::string& token) const;

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
