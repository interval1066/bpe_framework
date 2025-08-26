#include "lm/models/transformer_block.hpp"
#include <iostream>

namespace lm {

TransformerBlock::TransformerBlock(size_t d_model, size_t num_heads, size_t d_ff, float dropout)
    : d_model_(d_model), num_heads_(num_heads), d_ff_(d_ff), dropout_(dropout) {
    
    // Initialize multi-head attention
    attention_ = std::make_unique<MultiHeadAttention>(d_model, num_heads, dropout);
    
    // Initialize feed-forward network
    feed_forward_ = std::make_unique<FeedForward>(d_model, d_ff, dropout);
    
    // Initialize layer normalization
    norm1_ = std::make_unique<LayerNorm>(d_model);
    norm2_ = std::make_unique<LayerNorm>(d_model);
    
    std::cout << "Initialized TransformerBlock with:\n";
    std::cout << "  d_model: " << d_model_ << "\n";
    std::cout << "  num_heads: " << num_heads_ << "\n";
    std::cout << "  d_ff: " << d_ff_ << "\n";
    std::cout << "  dropout: " << dropout_ << "\n";
}

std::vector<Tensor> TransformerBlock::parameters() const {
    std::vector<Tensor> params;
    
    // Add attention parameters
    auto attention_params = attention_->parameters();
    params.insert(params.end(), attention_params.begin(), attention_params.end());
    
    // Add feed-forward parameters
    auto ff_params = feed_forward_->parameters();
    params.insert(params.end(), ff_params.begin(), ff_params.end());
    
    // Add layer norm parameters
    auto norm1_params = norm1_->parameters();
    params.insert(params.end(), norm1_params.begin(), norm1_params.end());
    
    auto norm2_params = norm2_->parameters();
    params.insert(params.end(), norm2_params.begin(), norm2_params.end());
    
    return params;
}

void TransformerBlock::set_training(bool training) {
    training_ = training;
    attention_->set_training(training);
    feed_forward_->set_training(training);
}

Tensor TransformerBlock::forward(const Tensor& input, const Tensor& mask) const {
    // Self-attention with residual connection
    Tensor attention_output = attention_->forward(input, input, input, mask);
    Tensor norm1_output = norm1_->forward(input + attention_output);
    
    // Feed-forward with residual connection
    Tensor ff_output = feed_forward_->forward(norm1_output);
    Tensor output = norm2_->forward(norm1_output + ff_output);
    
    return output;
}

} // namespace lm
