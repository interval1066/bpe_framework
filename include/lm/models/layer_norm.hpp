#pragma once

#include "lm/core/tensor.hpp"
#include <vector>

namespace lm {

class LayerNorm {
public:
    LayerNorm(size_t d_model, float eps = 1e-5f);
    
    std::vector<Tensor> parameters() const;
    void set_training(bool training);
    Tensor forward(const Tensor& input);
    
private:
    size_t d_model_;
    float eps_;
    
    Tensor gamma_;
    Tensor beta_;
};

} // namespace lm
