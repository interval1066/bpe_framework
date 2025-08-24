#include "lm/models/transformer.hpp"
#include <iostream>
#include <random>

namespace lm {

Transformer::Transformer(size_t vocab_size, size_t d_model, size_t num_heads, 
                         size_t d_ff, size_t num_layers, size_t max_seq_len, float dropout)
    : vocab_size_(vocab_size), d_model_(d_model), num_heads_(num_heads),
      d_ff_(d_ff), num_layers_(num_layers), max_seq_len_(max_seq_len), dropout_(dropout) {
    
    // Initialize embedding layer
    embedding_ = Tensor::randn({vocab_size_, d_model_}, 0.0, 0.02);
    
    // Initialize positional encoding - FIXED: explicit vector creation
    positional_encoding_ = Tensor(std::vector<size_t>{max_seq_len_, d_model_});    for (size_t pos = 0; pos < max_seq_len_; ++pos) {
        for (size_t i = 0; i < d_model_; ++i) {
            if (i % 2 == 0) {
                positional_encoding_(pos, i) = std::sin(pos / std::pow(10000, 2.0 * i / d_model_));
            } else {
                positional_encoding_(pos, i) = std::cos(pos / std::pow(10000, 2.0 * (i - 1) / d_model_));
            }
        }
    }
    
    // Initialize transformer blocks
    for (size_t i = 0; i < num_layers_; ++i) {
        transformer_blocks_.push_back(std::make_unique<TransformerBlock>(d_model_, num_heads_, d_ff_, dropout_));
    }
    
    // Initialize output layer
    output_layer_ = Tensor::randn({d_model_, vocab_size_}, 0.0, 0.02);
    
    std::cout << "Initialized Transformer with:\n";
    std::cout << "  vocab_size: " << vocab_size_ << "\n";
    std::cout << "  d_model: " << d_model_ << "\n";
    std::cout << "  num_heads: " << num_heads_ << "\n";
    std::cout << "  d_ff: " << d_ff_ << "\n";
    std::cout << "  num_layers: " << num_layers_ << "\n";
    std::cout << "  max_seq_len: " << max_seq_len_ << "\n";
    std::cout << "  dropout: " << dropout_ << "\n";
}

std::vector<Tensor> Transformer::parameters() const {
    std::vector<Tensor> params;
    
    // Add embedding parameters
    params.push_back(embedding_);
    
    // Add transformer block parameters
    for (const auto& block : transformer_blocks_) {
        auto block_params = block->parameters();
        params.insert(params.end(), block_params.begin(), block_params.end());
    }
    
    // Add output layer parameters
    params.push_back(output_layer_);
    
    return params;
}

void Transformer::set_training(bool training) {
    training_ = training;
    
    // Set training mode for all transformer blocks
    for (auto& block : transformer_blocks_) {
        block->set_training(training);
    }
    
    std::cout << "Set training mode to: " << (training ? "true" : "false") << "\n";
}

Tensor Transformer::forward(const Tensor& input, const Tensor& mask) {
    // Get input dimensions
    size_t batch_size = input.shape()[0];
    size_t seq_len = input.shape()[1];
    
    // Convert token IDs to embeddings
    Tensor embeddings(std::vector<size_t>{batch_size, seq_len, d_model_});
    
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t t = 0; t < seq_len; ++t) {
            size_t token_id = static_cast<size_t>(input(b, t));
            if (token_id < vocab_size_) {
                for (size_t d = 0; d < d_model_; ++d) {
                    // Use 3D indexing
                    embeddings(b, t, d) = embedding_(token_id, d);
                }
            }
        }
    }
    
    // Add positional encoding
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t t = 0; t < seq_len; ++t) {
            for (size_t d = 0; d < d_model_; ++d) {
                // Use 3D indexing
                embeddings(b, t, d) += positional_encoding_(t, d);
            }
        }
    }
    
    // Apply dropout during training
    if (training_) {
        embeddings = apply_dropout(embeddings, dropout_);
    }
    
    // Pass through transformer blocks
    Tensor hidden_states = embeddings;
    for (auto& block : transformer_blocks_) {
        hidden_states = block->forward(hidden_states, mask);
    }
    
    // Apply output layer
    Tensor logits(std::vector<size_t>{batch_size, seq_len, vocab_size_});
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t t = 0; t < seq_len; ++t) {
            for (size_t v = 0; v < vocab_size_; ++v) {
                logits(b, t, v) = 0.0;
                for (size_t d = 0; d < d_model_; ++d) {
                    // Use 3D indexing
                    logits(b, t, v) += hidden_states(b, t, d) * output_layer_(d, v);
                }
            }
        }
    }
    
    return logits;
}

Tensor Transformer::apply_dropout(const Tensor& input, float dropout_rate) {
    if (dropout_rate <= 0.0) return input;
    
    Tensor output = input;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::bernoulli_distribution dist(1.0 - dropout_rate);
    
    for (size_t i = 0; i < output.size(); ++i) {
        if (!dist(gen)) {
            output(i) = 0.0;
        } else {
            output(i) /= (1.0 - dropout_rate);
        }
    }
    
    return output;
}

} // namespace lm
