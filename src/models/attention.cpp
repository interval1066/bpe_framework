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
    w_q_ = Tensor::xavier(std::vector<size_t>{d_model_, d_model_});
    w_k_ = Tensor::xavier(std::vector<size_t>{d_model_, d_model_});
    w_v_ = Tensor::xavier(std::vector<size_t>{d_model_, d_model_});
    w_o_ = Tensor::xavier(std::vector<size_t>{d_model_, d_model_});
    
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
    //size_t batch_size = query.shape()[0];
    //size_t seq_len = query.shape()[1];
    
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
    Tensor result(std::vector<size_t>{batch_size, seq_len, num_heads_, d_k_});
    
    // Calculate strides for flat indexing
    size_t x_stride_1 = d_model_;        // stride for sequence position in x
    size_t result_stride_1 = num_heads_ * d_k_;  // stride for sequence position in result
    size_t result_stride_2 = d_k_;               // stride for head position in result
    
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t t = 0; t < seq_len; ++t) {
            for (size_t h = 0; h < num_heads_; ++h) {
                for (size_t d = 0; d < d_k_; ++d) {
                    size_t src_idx = d + h * d_k_;
                    
                    // Calculate flat indices
                    size_t x_index = b * seq_len * x_stride_1 + t * x_stride_1 + src_idx;
                    size_t result_index = b * seq_len * result_stride_1 + 
                                         t * result_stride_1 + 
                                         h * result_stride_2 + 
                                         d;
                    
                    result(result_index) = x(x_index);
                }
            }
        }
    }
    
    // Transpose to [batch_size, num_heads, seq_len, d_k]
    Tensor transposed(std::vector<size_t>{batch_size, num_heads_, seq_len, d_k_});
    
    // Calculate strides for transposed tensor
    size_t transposed_stride_1 = seq_len * d_k_;  // stride for head position
    size_t transposed_stride_2 = d_k_;            // stride for sequence position
    
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t h = 0; h < num_heads_; ++h) {
            for (size_t t = 0; t < seq_len; ++t) {
                for (size_t d = 0; d < d_k_; ++d) {
                    // Calculate flat indices
                    size_t result_index = b * seq_len * result_stride_1 + 
                                         t * result_stride_1 + 
                                         h * result_stride_2 + 
                                         d;
                    size_t transposed_index = b * num_heads_ * transposed_stride_1 + 
                                            h * transposed_stride_1 + 
                                            t * transposed_stride_2 + 
                                            d;
                    
                    transposed(transposed_index) = result(result_index);
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
    Tensor transposed(std::vector<size_t>{batch_size, seq_len, num_heads, d_k});
    
    // Calculate strides for flat indexing
    size_t x_stride_1 = seq_len * d_k;  // stride for head position in x
    size_t x_stride_2 = d_k;            // stride for sequence position in x
    size_t transposed_stride_1 = num_heads * d_k;  // stride for sequence position in transposed
    size_t transposed_stride_2 = d_k;              // stride for head position in transposed
    
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t t = 0; t < seq_len; ++t) {
            for (size_t h = 0; h < num_heads; ++h) {
                for (size_t d = 0; d < d_k; ++d) {
                    // Calculate flat indices
                    size_t x_index = b * num_heads * x_stride_1 + 
                                    h * x_stride_1 + 
                                    t * x_stride_2 + 
                                    d;
                    size_t transposed_index = b * seq_len * transposed_stride_1 + 
                                            t * transposed_stride_1 + 
                                            h * transposed_stride_2 + 
                                            d;
                    
                    transposed(transposed_index) = x(x_index);
                }
            }
        }
    }
    
    // Combine to [batch_size, seq_len, d_model]
    Tensor result(std::vector<size_t>{batch_size, seq_len, d_model_});
    
    // Calculate strides for result
    size_t result_stride_1 = d_model_;  // stride for sequence position
    //size_t result_stride_2 = d_k;       // stride for head position
    
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t t = 0; t < seq_len; ++t) {
            for (size_t h = 0; h < num_heads; ++h) {
                for (size_t d = 0; d < d_k; ++d) {
                    // Calculate flat index for transposed
                    size_t transposed_index = b * seq_len * transposed_stride_1 + 
                                            t * transposed_stride_1 + 
                                            h * transposed_stride_2 + 
                                            d;
                    
                    // Calculate destination index in result
                    size_t dst_idx = d + h * d_k;
                    
                    // Calculate flat index for result
                    size_t result_index = b * seq_len * result_stride_1 + 
                                         t * result_stride_1 + 
                                         dst_idx;
                    
                    result(result_index) = transposed(transposed_index);
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
    Tensor scores(std::vector<size_t>{batch_size, num_heads, seq_len, seq_len});
    
    // Calculate strides for flat indexing
    size_t q_stride_1 = seq_len * d_k;  // stride for head position in q
    size_t q_stride_2 = d_k;            // stride for sequence position in q
    size_t k_stride_1 = seq_len * d_k;  // stride for head position in k
    size_t k_stride_2 = d_k;            // stride for sequence position in k
    size_t scores_stride_1 = seq_len * seq_len;  // stride for head position in scores
    size_t scores_stride_2 = seq_len;            // stride for sequence position in scores
    
    // Matrix multiplication: q * k^T
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t h = 0; h < num_heads; ++h) {
            for (size_t i = 0; i < seq_len; ++i) {
                for (size_t j = 0; j < seq_len; ++j) {
                    // Calculate flat index for scores
                    size_t scores_index = b * num_heads * scores_stride_1 + 
                                         h * scores_stride_1 + 
                                         i * scores_stride_2 + 
                                         j;
                    
                    scores(scores_index) = 0.0;
                    
                    for (size_t d = 0; d < d_k; ++d) {
                        // Calculate flat indices for q and k
                        size_t q_index = b * num_heads * q_stride_1 + 
                                        h * q_stride_1 + 
                                        i * q_stride_2 + 
                                        d;
                        size_t k_index = b * num_heads * k_stride_1 + 
                                        h * k_stride_1 + 
                                        j * k_stride_2 + 
                                        d;
                        
                        scores(scores_index) += q(q_index) * k(k_index);
                    }
                    
                    scores(scores_index) /= std::sqrt(static_cast<float>(d_k));
                }
            }
        }
    }
    
    // Apply mask if provided
    if (mask.size() > 0) {
        size_t mask_stride_1 = seq_len * seq_len;  // stride for batch position in mask
        size_t mask_stride_2 = seq_len;            // stride for sequence position in mask
        
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t h = 0; h < num_heads; ++h) {
                for (size_t i = 0; i < seq_len; ++i) {
                    for (size_t j = 0; j < seq_len; ++j) {
                        // Calculate flat indices
                        size_t scores_index = b * num_heads * scores_stride_1 + 
                                             h * scores_stride_1 + 
                                             i * scores_stride_2 + 
                                             j;
                        size_t mask_index = b * mask_stride_1 + 
                                           i * mask_stride_2 + 
                                           j;
                        
                        if (mask(mask_index) == 0.0) {
                            scores(scores_index) = -1e9; // Large negative value
                        }
                    }
                }
            }
        }
    }
    
    // Apply softmax to get attention weights
    Tensor weights(std::vector<size_t>{batch_size, num_heads, seq_len, seq_len});
    
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t h = 0; h < num_heads; ++h) {
            for (size_t i = 0; i < seq_len; ++i) {
                // Find max for numerical stability
                float max_val = -std::numeric_limits<float>::infinity();
                for (size_t j = 0; j < seq_len; ++j) {
                    size_t scores_index = b * num_heads * scores_stride_1 + 
                                         h * scores_stride_1 + 
                                         i * scores_stride_2 + 
                                         j;
                    if (scores(scores_index) > max_val) {
                        max_val = scores(scores_index);
                    }
                }
                
                // Compute exponentials and sum
                float sum = 0.0;
                for (size_t j = 0; j < seq_len; ++j) {
                    size_t scores_index = b * num_heads * scores_stride_1 + 
                                         h * scores_stride_1 + 
                                         i * scores_stride_2 + 
                                         j;
                    size_t weights_index = b * num_heads * scores_stride_1 + 
                                          h * scores_stride_1 + 
                                          i * scores_stride_2 + 
                                          j;
                    
                    weights(weights_index) = std::exp(scores(scores_index) - max_val);
                    sum += weights(weights_index);
                }
                
                // Normalize
                for (size_t j = 0; j < seq_len; ++j) {
                    size_t weights_index = b * num_heads * scores_stride_1 + 
                                          h * scores_stride_1 + 
                                          i * scores_stride_2 + 
                                          j;
                    
                    weights(weights_index) /= sum;
                }
            }
        }
    }
    
    // Apply dropout during training
    if (training_) {
        weights = apply_dropout(weights, dropout_);
    }
    
    // Multiply weights by values
    Tensor output(std::vector<size_t>{batch_size, num_heads, seq_len, d_k});
    
    // Calculate strides for output and v
    size_t output_stride_1 = seq_len * d_k;  // stride for head position in output
    size_t output_stride_2 = d_k;            // stride for sequence position in output
    size_t v_stride_1 = seq_len * d_k;       // stride for head position in v
    size_t v_stride_2 = d_k;                 // stride for sequence position in v
    
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t h = 0; h < num_heads; ++h) {
            for (size_t i = 0; i < seq_len; ++i) {
                for (size_t d = 0; d < d_k; ++d) {
                    // Calculate flat index for output
                    size_t output_index = b * num_heads * output_stride_1 + 
                                         h * output_stride_1 + 
                                         i * output_stride_2 + 
                                         d;
                    
                    output(output_index) = 0.0;
                    
                    for (size_t j = 0; j < seq_len; ++j) {
                        // Calculate flat indices for weights and v
                        size_t weights_index = b * num_heads * scores_stride_1 + 
                                              h * scores_stride_1 + 
                                              i * scores_stride_2 + 
                                              j;
                        size_t v_index = b * num_heads * v_stride_1 + 
                                        h * v_stride_1 + 
                                        j * v_stride_2 + 
                                        d;
                        
                        output(output_index) += weights(weights_index) * v(v_index);
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
