#pragma once

#include "lm/core/tensor.hpp"
#include "lm/models/transformer_block.hpp"
#include <vector>
#include <memory>
#include <cmath>

namespace lm {

class Transformer {
public:
    Transformer(size_t vocab_size, size_t d_model, size_t num_heads, 
                size_t d_ff, size_t num_layers, size_t max_seq_len, float dropout = 0.1f);
    
    std::vector<Tensor> parameters() const;
    void set_training(bool training);
    Tensor forward(const Tensor& input, const Tensor& mask = Tensor());
    
private:
    Tensor apply_dropout(const Tensor& input, float dropout_rate);
    
    size_t vocab_size_, d_model_, num_heads_, d_ff_, num_layers_, max_seq_len_;
    float dropout_;
    bool training_ = false;
    
    Tensor embedding_;
    Tensor positional_encoding_;
    Tensor output_layer_;
    std::vector<std::unique_ptr<TransformerBlock>> transformer_blocks_;
};

} // namespace lm
