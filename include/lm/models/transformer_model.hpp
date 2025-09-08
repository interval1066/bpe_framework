// transformer_model.hpp
#pragma once

#include <vector>
#include <cstdint>
#include <memory>
#include <cmath>
#include <random>
#include <iostream>
#include "lm/tokenizer/token_types.hpp"

namespace lm {

class TransformerModel {
public:
    TransformerModel(size_t vocab_size, 
                    size_t d_model = 512, 
                    size_t n_layers = 6, 
                    size_t n_heads = 8,
                    size_t d_ff = 2048,
                    float dropout = 0.1);
    
    ~TransformerModel();
    
    // Forward pass
    std::vector<float> forward(const std::vector<TokenID>& input_tokens);
    
    // Training methods
    void train_step(const std::vector<TokenID>& input_tokens, 
                   const std::vector<TokenID>& target_tokens);
    float calculate_loss(const std::vector<float>& logits, 
                        const std::vector<TokenID>& targets);
    
    // Generation methods
    std::vector<TokenID> generate(const std::vector<TokenID>& context, 
                                 size_t max_length = 100,
                                 float temperature = 1.0);
    
    // Serialization
    bool save(const std::string& filename);
    bool load(const std::string& filename);
    
    // Get model info
    size_t get_vocab_size() const { return vocab_size_; }
    size_t get_d_model() const { return d_model_; }

private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;
    
    // Model parameters
    size_t vocab_size_;
    size_t d_model_;
    size_t n_layers_;
    size_t n_heads_;
    size_t d_ff_;
    float dropout_;
};

} // namespace lm
