#pragma once

#include "lm/core/tensor.hpp"
#include <vector>

namespace lm {

class FeedForward {
public:
    FeedForward(size_t d_model, size_t d_ff, float dropout = 0.1f);
    
    std::vector<Tensor> parameters() const;
    void set_training(bool training);
    Tensor forward(const Tensor& input) const;
    
private:
    Tensor apply_dropout(const Tensor& input, float dropout_rate) const;
    Tensor gelu(const Tensor& input) const;
    
    size_t d_model_;
    size_t d_ff_;
    float dropout_;
    bool training_ = false;
    
    Tensor w1_;
    Tensor b1_;
    Tensor w2_;
    Tensor b2_;
};

} // namespace lm

