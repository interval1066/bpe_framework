// src/models/transformer_model.cpp
#include <lm/models/transformer_model.hpp>
#include <lm/core/eigen_serialization.hpp>
#include <eigen3/Eigen/Dense>
#include <vector>
#include <memory>
#include <random>
#include <cmath>
#include <algorithm>
#include <iostream>

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

// Helper function for layer normalization gradient
Eigen::VectorXf layer_norm_backward(const Eigen::VectorXf& x, 
                                   const Eigen::VectorXf& gamma,
                                   const Eigen::VectorXf& beta,
                                   const Eigen::VectorXf& grad_output,
                                   float eps = 1e-5) {
    size_t n = x.size();
    Eigen::VectorXf mean = x.array().mean() * Eigen::VectorXf::Ones(n);
    Eigen::VectorXf var = ((x.array() - mean.array()).square().sum() / n) * Eigen::VectorXf::Ones(n);
    
    Eigen::VectorXf x_hat = (x.array() - mean.array()) / (var.array() + eps).sqrt();
    
    Eigen::VectorXf dgamma = grad_output.array() * x_hat.array();
    Eigen::VectorXf dbeta = grad_output;
    
    Eigen::VectorXf dx_hat = grad_output.array() * gamma.array();
    Eigen::VectorXf dvar = dx_hat.array() * (x.array() - mean.array()) * (-0.5f) * 
                          (var.array() + eps).pow(-1.5f);
    Eigen::VectorXf dmean = dx_hat.array() * (-1.0f) / (var.array() + eps).sqrt() + 
                           dvar.array() * (-2.0f) * (x.array() - mean.array()) / n;
    
    Eigen::VectorXf dx = dx_hat.array() / (var.array() + eps).sqrt() + 
                        dvar.array() * 2.0f * (x.array() - mean.array()) / n + 
                        dmean.array() / n;
    
    return dx;
}

// Helper function for softmax gradient
Eigen::MatrixXf softmax_backward(const Eigen::MatrixXf& softmax_output, 
                                const Eigen::MatrixXf& grad_output) {
    size_t seq_len = softmax_output.rows();
    size_t vocab_size = softmax_output.cols();
    
    Eigen::MatrixXf grad_input(seq_len, vocab_size);
    
    for (size_t i = 0; i < seq_len; i++) {
        Eigen::VectorXf s = softmax_output.row(i);
        Eigen::MatrixXf jacobian = s * s.transpose();
        jacobian.diagonal() = s.array() * (1.0f - s.array());
        
        grad_input.row(i) = grad_output.row(i) * jacobian;
    }
    
    return grad_input;
}

// Helper function for GELU gradient
Eigen::MatrixXf gelu_backward(const Eigen::MatrixXf& x, 
                             const Eigen::MatrixXf& grad_output) {
    const float sqrt_2_over_pi = std::sqrt(2.0f / static_cast<float>(M_PI));
    const float coef = 0.044715f;
    
    Eigen::MatrixXf grad_input = grad_output;
    
    for (int i = 0; i < x.rows(); i++) {
        for (int j = 0; j < x.cols(); j++) {
            float x_val = x(i, j);
            float x_cubed = x_val * x_val * x_val;
            float inner = sqrt_2_over_pi * (x_val + coef * x_cubed);
            float tanh_inner = std::tanh(inner);
            float sech_squared = 1.0f - tanh_inner * tanh_inner;
            
            float derivative = 0.5f * tanh_inner + 
                0.5f * x_val * sech_squared * sqrt_2_over_pi * (1.0f + 3.0f * coef * x_val * x_val) +
                0.5f * (1.0f + tanh_inner);
            
            grad_input(i, j) *= derivative;
        }
    }
    
    return grad_input;
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
        
        // Gradients
        Eigen::MatrixXf w_q_grad, w_k_grad, w_v_grad, w_o_grad;
        Eigen::VectorXf attn_gamma_grad, attn_beta_grad;
        Eigen::MatrixXf w_ff1_grad, w_ff2_grad;
        Eigen::VectorXf ff_gamma_grad, ff_beta_grad;
    };
    
    std::vector<TransformerBlock> blocks;
    
    // Final layers
    Eigen::MatrixXf lm_head;
    Eigen::VectorXf final_gamma, final_beta;
    
    // Gradients for final layers
    Eigen::MatrixXf lm_head_grad;
    Eigen::VectorXf final_gamma_grad, final_beta_grad;
    
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
    
    // Storage for intermediate values needed for backpropagation
    struct ForwardCache {
        Eigen::MatrixXf attention_scores;
        Eigen::MatrixXf attention_weights;
        Eigen::MatrixXf attention_output;
        Eigen::MatrixXf layer_norm_input;
        Eigen::MatrixXf ff_input;
        Eigen::MatrixXf gelu_output;
        Eigen::MatrixXf pre_attention_norm;
        Eigen::MatrixXf pre_ff_norm;
    };
    
    std::vector<ForwardCache> forward_caches;
    Eigen::MatrixXf embeddings;
    Eigen::MatrixXf final_output;
    
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
            
            // Initialize gradients to zero
            block.w_q_grad = Eigen::MatrixXf::Zero(d_model, d_model);
            block.w_k_grad = Eigen::MatrixXf::Zero(d_model, d_model);
            block.w_v_grad = Eigen::MatrixXf::Zero(d_model, d_model);
            block.w_o_grad = Eigen::MatrixXf::Zero(d_model, d_model);
            block.attn_gamma_grad = Eigen::VectorXf::Zero(d_model);
            block.attn_beta_grad = Eigen::VectorXf::Zero(d_model);
            block.w_ff1_grad = Eigen::MatrixXf::Zero(d_model, d_ff);
            block.w_ff2_grad = Eigen::MatrixXf::Zero(d_ff, d_model);
            block.ff_gamma_grad = Eigen::VectorXf::Zero(d_model);
            block.ff_beta_grad = Eigen::VectorXf::Zero(d_model);
        }
        
        // Initialize final layers
        lm_head = Eigen::MatrixXf::Random(d_model, vocab_size) * 0.02;
        final_gamma = Eigen::VectorXf::Ones(d_model);
        final_beta = Eigen::VectorXf::Zero(d_model);
        
        // Initialize final layer gradients to zero
        lm_head_grad = Eigen::MatrixXf::Zero(d_model, vocab_size);
        final_gamma_grad = Eigen::VectorXf::Zero(d_model);
        final_beta_grad = Eigen::VectorXf::Zero(d_model);
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
        
        // Clear previous cache if training
        if (is_training) {
            forward_caches.resize(n_layers);
        }
        
        // Create token embeddings
        embeddings.resize(seq_len, d_model);
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
        for (size_t layer = 0; layer < n_layers; layer++) {
            auto& block = blocks[layer];
            auto& cache = forward_caches[layer];
            
            if (is_training) {
                cache.pre_attention_norm = x;
            }
            
            // Self-attention
            Eigen::MatrixXf attn_output = self_attention(x, block.w_q, block.w_k, 
                                                        block.w_v, block.w_o, is_training);
            
            if (is_training) {
                cache.attention_output = attn_output;
            }
            
            // Residual connection and layer norm
            x = x + attn_output;
            if (is_training) {
                cache.layer_norm_input = x;
            }
            
            for (size_t i = 0; i < seq_len; i++) {
                x.row(i) = layer_norm(x.row(i).transpose(), block.attn_gamma, 
                                     block.attn_beta).transpose();
            }
            
            if (is_training) {
                cache.pre_ff_norm = x;
            }
            
            // Feed-forward
            Eigen::MatrixXf ff_output = feed_forward(x, block.w_ff1, block.w_ff2, is_training);
            
            if (is_training) {
                cache.gelu_output = ff_output;
            }
            
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
        final_output = x * lm_head;
        
        // Convert to vector
        std::vector<float> result(final_output.data(), final_output.data() + final_output.size());
        return result;
    }
    
    void backward(const Eigen::MatrixXf& grad_output) {
        size_t seq_len = grad_output.rows();
        size_t vocab_size = grad_output.cols();
        
        // Gradient through final layer norm and LM head
        Eigen::MatrixXf grad_final = grad_output * lm_head.transpose();
        
        // Update LM head gradient
        lm_head_grad += final_output.transpose() * grad_output;
        
        // Backward through final layer norm
        Eigen::MatrixXf grad_ln_final(seq_len, d_model);
        for (size_t i = 0; i < seq_len; i++) {
            // Fixed array operations
            Eigen::ArrayXf row_array = final_output.row(i).array();
            float mean = row_array.mean();
            float std_dev = std::sqrt(row_array.square().mean() - mean * mean + 1e-5f);
            
            grad_ln_final.row(i) = layer_norm_backward(final_output.row(i).transpose(),
                                                      final_gamma, final_beta,
                                                      grad_final.row(i).transpose()).transpose();
            
            // Update final layer norm gradients
            final_gamma_grad.array() += grad_final.row(i).array() * ((row_array - mean) / std_dev);
            final_beta_grad.array() += grad_final.row(i).array();
        }
        
        // Backward through transformer blocks in reverse order
        Eigen::MatrixXf grad = grad_ln_final;
        
        for (int layer = n_layers - 1; layer >= 0; layer--) {
            auto& block = blocks[layer];
            auto& cache = forward_caches[layer];
            
            // Backward through final layer norm of this block
            Eigen::MatrixXf grad_ln2(seq_len, d_model);
            for (size_t i = 0; i < seq_len; i++) {
                // Fixed array operations
                Eigen::ArrayXf row_array = cache.pre_ff_norm.row(i).array();
                float mean = row_array.mean();
                float std_dev = std::sqrt(row_array.square().mean() - mean * mean + 1e-5f);
                
                grad_ln2.row(i) = layer_norm_backward(cache.pre_ff_norm.row(i).transpose(),
                                                     block.ff_gamma, block.ff_beta,
                                                     grad.row(i).transpose()).transpose();
                
                // Update layer norm gradients
                block.ff_gamma_grad.array() += grad.row(i).array() * ((row_array - mean) / std_dev);
                block.ff_beta_grad.array() += grad.row(i).array();
            }
            
            // Backward through feed-forward network
            Eigen::MatrixXf grad_ff = gelu_backward(cache.pre_ff_norm * block.w_ff1, grad_ln2);
            
            // Update feed-forward weights gradients
            block.w_ff2_grad += cache.gelu_output.transpose() * grad_ln2;
            block.w_ff1_grad += cache.pre_ff_norm.transpose() * grad_ff;
            
            // Backward through residual connection
            Eigen::MatrixXf grad_res1 = grad_ln2 + grad;
            
            // Backward through first layer norm
            Eigen::MatrixXf grad_ln1(seq_len, d_model);
            for (size_t i = 0; i < seq_len; i++) {
                // Fixed array operations
                Eigen::ArrayXf row_array = cache.pre_attention_norm.row(i).array();
                float mean = row_array.mean();
                float std_dev = std::sqrt(row_array.square().mean() - mean * mean + 1e-5f);
                
                grad_ln1.row(i) = layer_norm_backward(cache.pre_attention_norm.row(i).transpose(),
                                                     block.attn_gamma, block.attn_beta,
                                                     grad_res1.row(i).transpose()).transpose();
                
                // Update layer norm gradients
                block.attn_gamma_grad.array() += grad_res1.row(i).array() * ((row_array - mean) / std_dev);
                block.attn_beta_grad.array() += grad_res1.row(i).array();
            }
            
            // Backward through self-attention (simplified)
            // This is a complex operation that would need its own implementation
            // For now, we'll use a simplified approach
            
            // Update attention weights gradients (simplified)
            block.w_o_grad += cache.attention_output.transpose() * grad_ln1;
            block.w_v_grad += cache.pre_attention_norm.transpose() * grad_ln1;
            block.w_k_grad += cache.pre_attention_norm.transpose() * grad_ln1;
            block.w_q_grad += cache.pre_attention_norm.transpose() * grad_ln1;
            
            // Prepare gradient for next layer
            grad = grad_ln1;
        }
        
        // Backward through embeddings (simplified)
        // In a complete implementation, we would update token and position embeddings
    }
    
    void update_parameters(float learning_rate) {
        // Update final layers
        lm_head -= learning_rate * lm_head_grad;
        final_gamma -= learning_rate * final_gamma_grad;
        final_beta -= learning_rate * final_beta_grad;
        
        // Update transformer blocks
        for (auto& block : blocks) {
            block.w_q -= learning_rate * block.w_q_grad;
            block.w_k -= learning_rate * block.w_k_grad;
            block.w_v -= learning_rate * block.w_v_grad;
            block.w_o -= learning_rate * block.w_o_grad;
            block.attn_gamma -= learning_rate * block.attn_gamma_grad;
            block.attn_beta -= learning_rate * block.attn_beta_grad;
            block.w_ff1 -= learning_rate * block.w_ff1_grad;
            block.w_ff2 -= learning_rate * block.w_ff2_grad;
            block.ff_gamma -= learning_rate * block.ff_gamma_grad;
            block.ff_beta -= learning_rate * block.ff_beta_grad;
        }
        
        // Reset gradients
        zero_grad();
    }
    
    void zero_grad() {
        // Reset final layer gradients
        lm_head_grad.setZero();
        final_gamma_grad.setZero();
        final_beta_grad.setZero();
        
        // Reset transformer block gradients
        for (auto& block : blocks) {
            block.w_q_grad.setZero();
            block.w_k_grad.setZero();
            block.w_v_grad.setZero();
            block.w_o_grad.setZero();
            block.attn_gamma_grad.setZero();
            block.attn_beta_grad.setZero();
            block.w_ff1_grad.setZero();
            block.w_ff2_grad.setZero();
            block.ff_gamma_grad.setZero();
            block.ff_beta_grad.setZero();
        }
    }
};

// Factory methods
std::shared_ptr<TransformerModel> TransformerModel::create() {
    return std::shared_ptr<TransformerModel>(new TransformerModel());
}

std::shared_ptr<TransformerModel> TransformerModel::create(size_t vocab_size, size_t d_model, 
                                                         size_t n_layers, size_t n_heads, 
                                                         size_t d_ff, float dropout) {
    return std::shared_ptr<TransformerModel>(
        new TransformerModel(vocab_size, d_model, n_layers, n_heads, d_ff, dropout)
    );
}

// Private constructors
TransformerModel::TransformerModel() 
    : vocab_size_(0), d_model_(0), n_layers_(0), n_heads_(0), d_ff_(0), dropout_(0.0f),
      pimpl_(nullptr) {
}

TransformerModel::TransformerModel(size_t vocab_size, size_t d_model, 
                                 size_t n_layers, size_t n_heads, 
                                 size_t d_ff, float dropout)
    : vocab_size_(vocab_size), d_model_(d_model), n_layers_(n_layers),
      n_heads_(n_heads), d_ff_(d_ff), dropout_(dropout) {
    pimpl_ = std::make_unique<Impl>(vocab_size, d_model, n_layers, 
                                   n_heads, d_ff, dropout);
}

// Destructor - must be defined after Impl is defined
TransformerModel::~TransformerModel() = default;

// Public methods
std::vector<float> TransformerModel::forward(const std::vector<TokenID>& input_tokens) {
    return pimpl_->forward(input_tokens, false); // false for inference mode
}

void TransformerModel::backward(const std::vector<float>& grad_output) {
    size_t seq_len = pimpl_->final_output.rows();
    size_t vocab_size = pimpl_->final_output.cols();
    
    // Convert vector gradient to matrix
    Eigen::Map<const Eigen::MatrixXf> grad_output_map(grad_output.data(), seq_len, vocab_size);
    pimpl_->backward(grad_output_map);
}

void TransformerModel::train_step(const std::vector<TokenID>& input_tokens, 
                                const std::vector<TokenID>& target_tokens,
                                float learning_rate) {
    // Forward pass
    auto logits = pimpl_->forward(input_tokens, true); // true for training mode
    
    // Calculate loss and gradient
    float loss = calculate_loss(logits, target_tokens);
    
    // Compute gradient of loss with respect to logits
    size_t seq_len = target_tokens.size();
    size_t vocab_size = vocab_size_;
    std::vector<float> grad_output(seq_len * vocab_size, 0.0f);
    
    for (size_t i = 0; i < seq_len; i++) {
        // Softmax gradient: dL/dz = p - y (for cross-entropy loss)
        const float* pos_logits = &logits[i * vocab_size];
        
        // Softmax
        float max_logit = *std::max_element(pos_logits, pos_logits + vocab_size);
        float sum_exp = 0.0;
        for (size_t j = 0; j < vocab_size; j++) {
            sum_exp += std::exp(pos_logits[j] - max_logit);
        }
        
        // Compute gradient
        for (size_t j = 0; j < vocab_size; j++) {
            float p = std::exp(pos_logits[j] - max_logit) / sum_exp;
            float y = (j == target_tokens[i]) ? 1.0f : 0.0f;
            grad_output[i * vocab_size + j] = (p - y) / seq_len;
        }
    }
    
    // Backward pass
    backward(grad_output);
    
    // Update parameters
    pimpl_->update_parameters(learning_rate);
    
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

void TransformerModel::serialize(std::ostream& stream) const {
    cereal::BinaryOutputArchive archive(stream);
    
    // Serialize basic parameters
    archive(vocab_size_, d_model_, n_layers_, n_heads_, d_ff_, dropout_);
    
    // Serialize the Impl data
    if (pimpl_) {
        // Serialize embeddings
        archive(pimpl_->token_embedding, pimpl_->position_embedding);
        
        // Serialize transformer blocks
        archive(pimpl_->blocks.size());
        for (const auto& block : pimpl_->blocks) {
            archive(block.w_q, block.w_k, block.w_v, block.w_o);
            archive(block.attn_gamma, block.attn_beta);
            archive(block.w_ff1, block.w_ff2);
            archive(block.ff_gamma, block.ff_beta);
            archive(block.dropout_rate);
        }
        
        // Serialize final layers
        archive(pimpl_->lm_head, pimpl_->final_gamma, pimpl_->final_beta);
    }
}

void TransformerModel::deserialize(std::istream& stream) {
    cereal::BinaryInputArchive archive(stream);
    
    // Deserialize basic parameters
    archive(vocab_size_, d_model_, n_layers_, n_heads_, d_ff_, dropout_);
    
    // Create new Impl instance
    pimpl_ = std::make_unique<Impl>(vocab_size_, d_model_, n_layers_, 
                                   n_heads_, d_ff_, dropout_);
    
    // Deserialize the Impl data
    if (pimpl_) {
        // Deserialize embeddings
        archive(pimpl_->token_embedding, pimpl_->position_embedding);
        
        // Deserialize transformer blocks
        size_t num_blocks;
        archive(num_blocks);
        pimpl_->blocks.resize(num_blocks);
        for (auto& block : pimpl_->blocks) {
            archive(block.w_q, block.w_k, block.w_v, block.w_o);
            archive(block.attn_gamma, block.attn_beta);
            archive(block.w_ff1, block.w_ff2);
            archive(block.ff_gamma, block.ff_beta);
            archive(block.dropout_rate);
        }
        
        // Deserialize final layers
        archive(pimpl_->lm_head, pimpl_->final_gamma, pimpl_->final_beta);
    }
}

bool TransformerModel::save(const std::string& filename) {
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs.is_open()) {
        return false;
    }
    
    try {
        serialize(ofs);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error saving model: " << e.what() << std::endl;
        return false;
    }
}

bool TransformerModel::load(const std::string& filename) {
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs.is_open()) {
        return false;
    }
    
    try {
        deserialize(ifs);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return false;
    }
}

std::vector<Tensor*> TransformerModel::parameters() {
    // This would return all parameter tensors for the optimizer
    // For now, return an empty vector as this is a complex implementation
    return {};
}

void TransformerModel::zero_grad() {
    if (pimpl_) {
        pimpl_->zero_grad();
    }
}

} // namespace lm
