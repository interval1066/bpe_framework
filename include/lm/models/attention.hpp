#pragma once

#include "lm/core/tensor.hpp"
#include <vector>
#include <memory>

namespace lm {

class MultiHeadAttention {
public:
    MultiHeadAttention(size_t d_model, size_t num_heads, float dropout = 0.1f);
    
    std::vector<Tensor> parameters() const;
    void set_training(bool training);
    Tensor forward(const Tensor& query, const Tensor& key, const Tensor& value, 
                   const Tensor& mask = Tensor());
    
private:
    Tensor split_heads(const Tensor& x);
    Tensor combine_heads(const Tensor& x);
    Tensor scaled_dot_product_attention(const Tensor& q, const Tensor& k, 
                                        const Tensor& v, const Tensor& mask);
    Tensor apply_dropout(const Tensor& input, float dropout_rate);
    
    size_t d_model_;
    size_t num_heads_;
    size_t d_k_;
    float dropout_;
    bool training_ = false;
    
    Tensor w_q_;
    Tensor w_k_;
    Tensor w_v_;
    Tensor w_o_;
};

} // namespace lm
