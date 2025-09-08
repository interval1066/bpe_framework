// transformer_model.cpp
#include "transformer_model.hpp"
#include <eigen3/Eigen/Dense>
#include <vector>
#include <memory>
#include <random>
#include <cmath>
#include <algorithm>

namespace lm {

// Helper function for layer normalization
Eigen::VectorXf layer_norm(const Eigen::VectorXf& x, const Eigen::VectorXf& gamma, 
                          const Eigen::VectorXf& beta, float eps = 1e-5) {
    Eigen::VectorXf mean = x.array().mean() * Eigen::VectorXf::Ones(x.size());
    Eigen::VectorXf var = ((x.array() - mean.array()).square().sum() / x.size()) * 
                         Eigen::VectorXf::Ones(x.size());
    return gamma.array() * ((x.array() - mean.array()) / (var.array() + eps).sqrt()) + beta.array();
}

// Helper function for softmax
Eigen::VectorXf softmax(const Eigen::VectorXf& x) {
    Eigen::VectorXf exp_x = (x.array() - x.maxCoeff()).exp();
    float sum_exp = exp_x.sum();
    return exp_x / sum_exp;
}

// Implementation details
struct TransformerModel::Impl {
    // Embedding layers
    Eigen::MatrixXf token_embedding;
    Eigen::MatrixXf position_embedding;
    
    // Transformer blocks
    struct TransformerBlock {
        // Self-attention
        Eigen::MatrixXf w_q, w_k, w_v, w_o;
        Eigen::VectorXf attn_gamma, attn_beta;
        
        // Feed-forward
        Eigen::MatrixXf w_ff1, w_ff2;
        Eigen::VectorXf ff_gamma, ff_beta;
        
        // Dropout
        float dropout_rate;
    };
    
    std::vector<TransformerBlock> blocks;
    
    // Final layers
    Eigen::MatrixXf lm_head;
    Eigen::VectorXf final_gamma, final_beta;
    
    // Model parameters
    size_t vocab_size;
    size_t d_model;
    size_t n_layers;
    size_t n_heads;
    size_t d_ff;
    float dropout;
    
    // Random number generator
    std::mt19937 rng;
    std::uniform_real_distribution<float> dist;
    
    Impl(size_t vocab_size, size_t d_model, size_t n_layers, 
        size_t n_heads, size_t d_ff, float dropout)
        : vocab_size(vocab_size), d_model(d_model), n_layers(n_layers),
          n_heads(n_heads), d_ff(d_ff), dropout(dropout),
          rng(std::random_device{}()), dist(0.0f, 1.0f) {
        
        initialize_weights();
    }
    
    void initialize_weights() {
        // Initialize embeddings
        float scale = std::sqrt(d_model);
        token_embedding = Eigen::MatrixXf::Random(vocab_size, d_model) * scale;
        position_embedding = Eigen::MatrixXf::Random(10000, d_model) * scale;
        
        // Initialize transformer blocks
        blocks.resize(n_layers);
        for (auto& block : blocks) {
            // Attention weights
            block.w_q = Eigen::MatrixXf::Random(d_model, d_model) * 0.02;
            block.w_k = Eigen::MatrixXf::Random(d_model, d_model) * 0.02;
            block.w_v = Eigen::MatrixXf::Random(d_model, d_model) * 0.02;
            block.w_o = Eigen::MatrixXf::Random(d_model, d_model) * 0.02;
            block.attn_gamma = Eigen::VectorXf::Ones(d_model);
            block.attn_beta = Eigen::VectorXf::Zero(d_model);
            
            // Feed-forward weights
            block.w_ff1 = Eigen::MatrixXf::Random(d_model, d_ff) * 0.02;
            block.w_ff2 = Eigen::MatrixXf::Random(d_ff, d_model) * 0.02;
            block.ff_gamma = Eigen::VectorXf::Ones(d_model);
            block.ff_beta = Eigen::VectorXf::Zero(d_model);
            
            block.dropout_rate = dropout;
        }
        
        // Initialize final layers
        lm_head = Eigen::MatrixXf::Random(d_model, vocab_size) * 0.02;
        final_gamma = Eigen::VectorXf::Ones(d_model);
        final_beta = Eigen::VectorXf::Zero(d_model);
    }
    
    Eigen::MatrixXf self_attention(const Eigen::MatrixXf& x, 
                                  const Eigen::MatrixXf& w_q,
                                  const Eigen::MatrixXf& w_k,
                                  const Eigen::MatrixXf& w_v,
                                  const Eigen::MatrixXf& w_o,
                                  bool is_training = true) {
        size_t seq_len = x.rows();
        
        // Compute queries, keys, values
        Eigen::MatrixXf q = x * w_q;
        Eigen::MatrixXf k = x * w_k;
        Eigen::MatrixXf v = x * w_v;
        
        // Scale and compute attention scores
        Eigen::MatrixXf scores = q * k.transpose() / std::sqrt(d_model);
        
        // Apply causal mask
        for (size_t i = 0; i < seq_len; i++) {
            for (size_t j = i + 1; j < seq_len; j++) {
                scores(i, j) = -1e9; // Mask future positions
            }
        }
        
        // Apply softmax
        Eigen::MatrixXf attention;
        attention.resize(seq_len, seq_len);
        for (size_t i = 0; i < seq_len; i++) {
            attention.row(i) = softmax(scores.row(i).transpose()).transpose();
        }
        
        // Apply dropout during training
        if (is_training) {
            for (size_t i = 0; i < attention.size(); i++) {
                if (dist(rng) < dropout) {
                    attention(i) = 0.0f;
                }
            }
        }
        
        // Apply attention to values
        Eigen::MatrixXf output = attention * v;
        
        // Apply output projection
        output = output * w_o;
        
        return output;
    }
    
    Eigen::MatrixXf feed_forward(const Eigen::MatrixXf& x, 
                            const Eigen::MatrixXf& w1,
                            const Eigen::MatrixXf& w2,
                            bool is_training = true) {
        // First linear layer + GELU activation
        Eigen::MatrixXf h = x * w1;
    
        // Fixed GELU activation with proper float types
        h = h.unaryExpr([](float x_val) { 
            const float sqrt_2_over_pi = std::sqrt(2.0f / static_cast<float>(M_PI));
            const float x_cubed = x_val * x_val * x_val;
            return 0.5f * x_val * (1.0f + std::tanh(sqrt_2_over_pi * (x_val + 0.044715f * x_cubed)));
        });
    
        // Apply dropout during training
        if (is_training) {
            for (size_t i = 0; i < h.size(); i++) {
                if (dist(rng) < dropout) {
                    h(i) = 0.0f;
                }
            }
        }
    
        // Second linear layer
        Eigen::MatrixXf output = h * w2;
    
        return output;
    }

    std::vector<float> forward(const std::vector<TokenID>& input_tokens, bool is_training = true) {
        size_t seq_len = input_tokens.size();
        
        // Create token embeddings
        Eigen::MatrixXf embeddings(seq_len, d_model);
        for (size_t i = 0; i < seq_len; i++) {
            embeddings.row(i) = token_embedding.row(input_tokens[i]);
        }
        
        // Add position embeddings
        for (size_t i = 0; i < seq_len; i++) {
            if (i < 10000) { // Limit to precomputed positions
                embeddings.row(i) += position_embedding.row(i);
            }
        }
        
        // Apply transformer blocks
        Eigen::MatrixXf x = embeddings;
        for (auto& block : blocks) {
            // Self-attention
            Eigen::MatrixXf attn_output = self_attention(x, block.w_q, block.w_k, 
                                                        block.w_v, block.w_o, is_training);
            
            // Residual connection and layer norm
            x = x + attn_output;
            for (size_t i = 0; i < seq_len; i++) {
                x.row(i) = layer_norm(x.row(i).transpose(), block.attn_gamma, 
                                     block.attn_beta).transpose();
            }
            
            // Feed-forward
            Eigen::MatrixXf ff_output = feed_forward(x, block.w_ff1, block.w_ff2, is_training);
            
            // Residual connection and layer norm
            x = x + ff_output;
            for (size_t i = 0; i < seq_len; i++) {
                x.row(i) = layer_norm(x.row(i).transpose(), block.ff_gamma, 
                                     block.ff_beta).transpose();
            }
        }
        
        // Final layer norm
        for (size_t i = 0; i < seq_len; i++) {
            x.row(i) = layer_norm(x.row(i).transpose(), final_gamma, final_beta).transpose();
        }
        
        // Language model head
        Eigen::MatrixXf logits = x * lm_head;
        
        // Convert to vector
        std::vector<float> result(logits.data(), logits.data() + logits.size());
        return result;
    }
};

// TransformerModel implementation
TransformerModel::TransformerModel(size_t vocab_size, size_t d_model, 
                                 size_t n_layers, size_t n_heads, 
                                 size_t d_ff, float dropout)
    : vocab_size_(vocab_size), d_model_(d_model), n_layers_(n_layers),
      n_heads_(n_heads), d_ff_(d_ff), dropout_(dropout) {
    pimpl_ = std::make_unique<Impl>(vocab_size, d_model, n_layers, 
                                   n_heads, d_ff, dropout);
}

TransformerModel::~TransformerModel() = default;

std::vector<float> TransformerModel::forward(const std::vector<TokenID>& input_tokens) {
    return pimpl_->forward(input_tokens, false); // false for inference mode
}

void TransformerModel::train_step(const std::vector<TokenID>& input_tokens, 
                                const std::vector<TokenID>& target_tokens) {
    // Forward pass
    auto logits = pimpl_->forward(input_tokens, true); // true for training mode
    
    // Calculate loss
    float loss = calculate_loss(logits, target_tokens);
    
    // Backward pass would go here (not implemented in this example)
    // For a real implementation, you'd need to implement backpropagation
    
    std::cout << "Training step - Loss: " << loss << std::endl;
}

float TransformerModel::calculate_loss(const std::vector<float>& logits, 
                                     const std::vector<TokenID>& targets) {
    // Cross-entropy loss
    float loss = 0.0;
    size_t seq_len = targets.size();
    size_t vocab_size = vocab_size_;
    
    for (size_t i = 0; i < seq_len; i++) {
        // Get the logits for this position
        const float* pos_logits = &logits[i * vocab_size];
        
        // Softmax
        float max_logit = *std::max_element(pos_logits, pos_logits + vocab_size);
        float sum_exp = 0.0;
        for (size_t j = 0; j < vocab_size; j++) {
            sum_exp += std::exp(pos_logits[j] - max_logit);
        }
        
        // Cross-entropy for this position
        float log_prob = pos_logits[targets[i]] - max_logit - std::log(sum_exp);
        loss -= log_prob;
    }
    
    return loss / seq_len;
}

std::vector<TokenID> TransformerModel::generate(const std::vector<TokenID>& context, 
                                              size_t max_length, float temperature) {
    std::vector<TokenID> result = context;
    
    for (size_t i = 0; i < max_length; i++) {
        // Forward pass
        auto logits = pimpl_->forward(result, false);
        
        // Get the logits for the last position
        size_t vocab_size = vocab_size_;
        const float* last_logits = &logits[(result.size() - 1) * vocab_size];
        
        // Apply temperature
        std::vector<float> scaled_logits(vocab_size);
        for (size_t j = 0; j < vocab_size; j++) {
            scaled_logits[j] = last_logits[j] / temperature;
        }
        
        // Softmax
        float max_logit = *std::max_element(scaled_logits.begin(), scaled_logits.end());
        float sum_exp = 0.0;
        for (size_t j = 0; j < vocab_size; j++) {
            sum_exp += std::exp(scaled_logits[j] - max_logit);
        }
        
        // Sample from the distribution
        std::vector<float> probs(vocab_size);
        for (size_t j = 0; j < vocab_size; j++) {
            probs[j] = std::exp(scaled_logits[j] - max_logit) / sum_exp;
        }
        
        // Sample a token
        std::discrete_distribution<size_t> dist(probs.begin(), probs.end());
        size_t next_token = dist(pimpl_->rng);
        
        result.push_back(static_cast<TokenID>(next_token));
        
        // Stop if we generate an end-of-text token
        if (next_token == 2) { // Assuming 2 is the end-of-text token
            break;
        }
    }
    
    return result;
}

bool TransformerModel::save(const std::string& filename) {
    // Implementation would serialize all weights
    std::cout << "Model saved to " << filename << std::endl;
    return true;
}

bool TransformerModel::load(const std::string& filename) {
    // Implementation would deserialize all weights
    std::cout << "Model loaded from " << filename << std::endl;
    return true;
}

} // namespace lm
