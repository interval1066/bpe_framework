#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include "token_types.hpp"

namespace lm {

class BPETokenizer {
public:
    BPETokenizer();
    ~BPETokenizer();

    // Training methods
    void train(const std::vector<std::string>& corpus, size_t vocab_size);
    
    // Encoding/decoding methods
    std::vector<TokenID> encode(const std::string& text) const;
    std::string decode(const std::vector<TokenID>& tokens) const;
    
    // Vocabulary methods
    size_t vocab_size() const;
    
    // Serialization methods
    bool save(const std::string& filename) const;
    bool load(const std::string& filename);
    
    // Special token methods
    TokenID eos_token_id() const;
    void set_eos_token_id(TokenID id);
    
    TokenID pad_token_id() const;
    void set_pad_token_id(TokenID id);
    
    TokenID unk_token_id() const;
    void set_unk_token_id(TokenID id);

    // Add special tokens to vocabulary
    void add_special_token(const std::string& token, TokenID id);
    
    // UTF-8 validation method
    //bool is_valid_utf8_asm(const char* str, size_t length);

    // Debug methods
    void enable_debug_logging(bool enable);
    void dump_vocabulary() const;
    void dump_merges() const;

    // Configuration methods for literary text handling
    void set_training_parameters(double stability_factor = 0.5, 
                                bool aggressive_normalization = true,
                                size_t min_sequence_length = 3,
                                bool preserve_paragraphs = true,
                                bool preserve_punctuation = true);
    
    // Text processing configuration
    void set_preserve_paragraphs(bool preserve);
    void set_preserve_punctuation(bool preserve);
    void set_handle_contractions(bool handle);
    
    // Getters for current configuration
    bool preserves_paragraphs() const;
    bool preserves_punctuation() const;
    bool handles_contractions() const;

private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};

} // namespace lm

