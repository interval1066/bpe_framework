#include "lm/models/attention.hpp"
#include <cmath>
#include <iostream>
#include <random>

namespace lm {

MultiHeadAttention::MultiHeadAttention(size_t d_model, size_t num_heads, float dropout)
    : d_model_(d_model), num_heads_(num_heads), dropout_(dropout) {
    
    // Ensure d_model is divisible by num_heads
    if (d_model % num_heads != 0) {
        throw std::invalid_argument("d_model must be divisible by num_heads");
    }
    
    d_k_ = d_model / num_heads;
    
    // Initialize weight matrices
    w_q_ = Tensor::xavier({d_model_, d_model_});
    w_k_ = Tensor::xavier({d_model_, d_model_});
    w_v_ = Tensor::xavier({d_model_, d_model_});
    w_o_ = Tensor::xavier({d_model_, d_model_});
    
    std::cout << "Initialized MultiHeadAttention with:\n";
    std::cout << "  d_model: " << d_model_ << "\n";
    std::cout << "  num_heads: " << num_heads_ << "\n";
    std::cout << "  d_k: " << d_k_ << "\n";
    std::cout << "  dropout: " << dropout_ << "\n";
}

std::vector<Tensor> MultiHeadAttention::parameters() const {
    return {w_q_, w_k_, w_v_, w_o_};
}

void MultiHeadAttention::set_training(bool training) {
    training_ = training;
}

Tensor MultiHeadAttention::forward(const Tensor& query, const Tensor& key, 
                                  const Tensor& value, const Tensor& mask) {
    // Get batch size and sequence length
    size_t batch_size = query.shape()[0];
    size_t seq_len = query.shape()[1];
    
    // Linear projections
    Tensor q = query.matmul(w_q_);  // [batch_size, seq_len, d_model]
    Tensor k = key.matmul(w_k_);    // [batch_size, seq_len, d_model]
    Tensor v = value.matmul(w_v_);  // [batch_size, seq_len, d_model]
    
    // Split into multiple heads
    q = split_heads(q);  // [batch_size, num_heads, seq_len, d_k]
    k = split_heads(k);  // [batch_size, num_heads, seq_len, d_k]
    v = split_heads(v);  // [batch_size, num_heads, seq_len, d_k]
    
    // Apply scaled dot-product attention
    Tensor attention_output = scaled_dot_product_attention(q, k, v, mask);
    
    // Combine heads
    attention_output = combine_heads(attention_output);  // [batch_size, seq_len, d_model]
    
    // Final linear projection
    Tensor output = attention_output.matmul(w_o_);  // [batch_size, seq_len, d_model]
    
    return output;
}

Tensor MultiHeadAttention::split_heads(const Tensor& x) {
    // x shape: [batch_size, seq_len, d_model]
    size_t batch_size = x.shape()[0];
    size_t seq_len = x.shape()[1];
    
    // Reshape to [batch_size, seq_len, num_heads, d_k]
    Tensor result({batch_size, seq_len, num_heads_, d_k_});
    
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t t = 0; t < seq_len; ++t) {
            for (size_t h = 0; h < num_heads_; ++h) {
                for (size_t d = 0; d < d_k_; ++d) {
                    size_t src_idx = d + h * d_k_;
                    result(b, t, h, d) = x(b, t, src_idx);
                }
            }
        }
    }
    
    // Transpose to [batch_size, num_heads, seq_len, d_k]
    Tensor transposed({batch_size, num_heads_, seq_len, d_k_});
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t h = 0; h < num_heads_; ++h) {
            for (size_t t = 0; t < seq_len; ++t) {
                for (size_t d = 0; d < d_k_; ++d) {
                    transposed(b, h, t, d) = result(b, t, h, d);
                }
            }
        }
    }
    
    return transposed;
}

Tensor MultiHeadAttention::combine_heads(const Tensor& x) {
    // x shape: [batch_size, num_heads, seq_len, d_k]
    size_t batch_size = x.shape()[0];
    size_t num_heads = x.shape()[1];
    size_t seq_len = x.shape()[2];
    size_t d_k = x.shape()[3];
    
    // Transpose back to [batch_size, seq_len, num_heads, d_k]
    Tensor transposed({batch_size, seq_len, num_heads, d_k});
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t t = 0; t < seq_len; ++t) {
            for (size_t h = 0; h < num_heads; ++h) {
                for (size_t d = 0; d < d_k; ++d) {
                    transposed(b, t, h, d) = x(b, h, t, d);
                }
            }
        }
    }
    
    // Combine to [batch_size, seq_len, d_model]
    Tensor result({batch_size, seq_len, d_model_});
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t t = 0; t < seq_len; ++t) {
            for (size_t h = 0; h < num_heads; ++h) {
                for (size_t d = 0; d < d_k; ++d) {
                    size_t dst_idx = d + h * d_k;
                    result(b, t, dst_idx) = transposed(b, t, h, d);
                }
            }
        }
    }
    
    return result;
}

Tensor MultiHeadAttention::scaled_dot_product_attention(const Tensor& q, const Tensor& k, 
                                                       const Tensor& v, const Tensor& mask) {
    // q, k, v shapes: [batch_size, num_heads, seq_len, d_k]
    size_t batch_size = q.shape()[0];
    size_t num_heads = q.shape()[1];
    size_t seq_len = q.shape()[2];
    size_t d_k = q.shape()[3];
    
    // Compute attention scores
    Tensor scores({batch_size, num_heads, seq_len, seq_len});
    
    // Matrix multiplication: q * k^T
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t h = 0; h < num_heads; ++h) {
            for (size_t i = 0; i < seq_len; ++i) {
                for (size_t j = 0; j < seq_len; ++j) {
                    scores(b, h, i, j) = 0.0;
                    for (size_t d = 0; d < d_k; ++d) {
                        scores(b, h, i, j) += q(b, h, i, d) * k(b, h, j, d);
                    }
                    scores(b, h, i, j) /= std::sqrt(static_cast<float>(d_k));
                }
            }
        }
    }
    
    // Apply mask if provided
    if (mask.size() > 0) {
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t h = 0; h < num_heads; ++h) {
                for (size_t i = 0; i < seq_len; ++i) {
                    for (size_t j = 0; j < seq_len; ++j) {
                        if (mask(b, i, j) == 0.0) {
                            scores(b, h, i, j) = -1e9; // Large negative value
                        }
                    }
                }
            }
        }
    }
    
    // Apply softmax to get attention weights
    Tensor weights({batch_size, num_heads, seq_len, seq_len});
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t h = 0; h < num_heads; ++h) {
            for (size_t i = 0; i < seq_len; ++i) {
                // Find max for numerical stability
                float max_val = scores(b, h, i, 0);
                for (size_t j = 1; j < seq_len; ++j) {
                    if (scores(b, h, i, j) > max_val) {
                        max_val = scores(b, h, i, j);
                    }
                }
                
                // Compute exponentials and sum
                float sum = 0.0;
                for (size_t j = 0; j < seq_len; ++j) {
                    weights(b, h, i, j) = std::exp(scores(b, h, i, j) - max_val);
                    sum += weights(b, h, i, j);
                }
                
                // Normalize
                for (size_t j = 0; j < seq_len; ++j) {
                    weights(b, h, i, j) /= sum;
                }
            }
        }
    }
    
    // Apply dropout during training
    if (training_) {
        weights = apply_dropout(weights, dropout_);
    }
    
    // Multiply weights by values
    Tensor output({batch_size, num_heads, seq_len, d_k});
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t h = 0; h < num_heads; ++h) {
            for (size_t i = 0; i < seq_len; ++i) {
                for (size_t d = 0; d < d_k; ++d) {
                    output(b, h, i, d) = 0.0;
                    for (size_t j = 0; j < seq_len; ++j) {
                        output(b, h, i, d) += weights(b, h, i, j) * v(b, h, j, d);
                    }
                }
            }
        }
    }
    
    return output;
}

Tensor MultiHeadAttention::apply_dropout(const Tensor& input, float dropout_rate) {
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
