#pragma once

#include "../core/tensor.hpp"
#include <vector>

namespace lm {

class AdamOptimizer {
public:
    AdamOptimizer(float learning_rate = 0.001, float beta1 = 0.9, float beta2 = 0.999, float epsilon = 1e-8);
    void step(std::vector<Tensor>& parameters);  // Remove const
    void zero_grad(std::vector<Tensor>& parameters);  // Remove const
    
    // Add getter for learning rate
    float get_learning_rate() const { return learning_rate_; }
    
private:
    float learning_rate_, beta1_, beta2_, epsilon_;
    int timestep_;
    std::vector<Tensor> m_, v_;  // First and second moment estimates
};

} // namespace lm
