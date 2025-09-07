// src/models/transformer.cpp
#include "lm/models/transformer.hpp"

namespace lm {

Transformer::Transformer(size_t vocab_size, size_t d_model, size_t num_heads, 
                       size_t d_ff, size_t num_layers, float dropout, size_t max_seq_length)
    : vocab_size_(vocab_size), d_model_(d_model), num_heads_(num_heads),
      d_ff_(d_ff), num_layers_(num_layers), dropout_(dropout), 
      max_seq_length_(max_seq_length),
      // Initialize LayerNorm with d_model
      final_layer_norm_(d_model) {
    
    // Initialize embeddings
    token_embeddings_ = Tensor::randn({vocab_size, d_model}, 0.0f, 0.02f, true);
    positional_embeddings_ = Tensor::randn({max_seq_length, d_model}, 0.0f, 0.02f, true);
    
    // Initialize transformer blocks
    for (size_t i = 0; i < num_layers; i++) {
        layers_.emplace_back(d_model, num_heads, d_ff, dropout);
    }
    
    // Initialize output projection
    output_projection_ = Tensor::randn({d_model, vocab_size}, 0.0f, 0.02f, true);
}

std::vector<Tensor> Transformer::get_parameters() const {
    // Return all model parameters
    std::vector<Tensor> params;
    // Add all parameters to the vector
    return params;
}

void Transformer::set_parameters(const std::vector<Tensor>& params) {
    // Set all model parameters
}

Tensor Transformer::forward(const std::vector<TokenID>& input) {
    // Implement forward pass for inference
    Tensor logits;
    // ... implementation
    return logits;
}

Tensor Transformer::forward(const std::vector<TokenID>& input, 
                          const std::vector<TokenID>& targets) {
    // Implement forward pass for training (with loss calculation)
    Tensor loss;
    // ... implementation
    return loss;
}

/*size_t Transformer::get_vocab_size() const {
    return vocab_size_;
}

size_t Transformer::get_max_sequence_length() const {
    return max_seq_length_;
}*/

void Transformer::save(const std::string& path) const {
    // Implement model serialization
}

void Transformer::load(const std::string& path) {
    // Implement model deserialization
}

} // namespace lm

