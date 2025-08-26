#pragma once

#include "lm/core/tensor.hpp"
#include "lm/models/attention.hpp"
#include "lm/models/feed_forward.hpp"
#include "lm/models/layer_norm.hpp"
#include <memory>
#include <vector>

namespace lm {

class TransformerBlock {
public:
    TransformerBlock(size_t d_model, size_t num_heads, size_t d_ff, float dropout);
    
    std::vector<Tensor> parameters() const;
    void set_training(bool training);
    Tensor forward(const Tensor& input, const Tensor& mask = Tensor()) const;
    
private:
    size_t d_model_, num_heads_, d_ff_;
    float dropout_;
    bool training_ = false;
    
    std::unique_ptr<MultiHeadAttention> attention_;
    std::unique_ptr<FeedForward> feed_forward_;
    std::unique_ptr<LayerNorm> norm1_;
    std::unique_ptr<LayerNorm> norm2_;
};

} // namespace lm

