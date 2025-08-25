#pragma once

#include "../core/tensor.hpp"
#include <vector>

namespace lm {

class AdamOptimizer {
public:
    AdamOptimizer(float learning_rate = 0.001, float beta1 = 0.9, float beta2 = 0.999, float epsilon = 1e-8);
    void step(const std::vector<Tensor>& parameters);
    void zero_grad(const std::vector<Tensor>& parameters);
    
private:
    float learning_rate_, beta1_, beta2_, epsilon_;
    int timestep_;
    std::vector<Tensor> m_, v_;  // First and second moment estimates
};

} // namespace lm
