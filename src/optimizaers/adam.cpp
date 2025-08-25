#include "lm/optimizers/adam.hpp"
#include <iostream>

namespace lm {

AdamOptimizer::AdamOptimizer(float learning_rate, float beta1, float beta2, float epsilon)
    : learning_rate_(learning_rate), beta1_(beta1), beta2_(beta2), epsilon_(epsilon), timestep_(0) {}

void AdamOptimizer::zero_grad(const std::vector<Tensor>& parameters) {
    for (auto& param : parameters) {
        if (param.requires_grad()) {
            param.zero_grad();
        }
    }
}

void AdamOptimizer::step(const std::vector<Tensor>& parameters) {
    timestep_++;
    
    for (size_t i = 0; i < parameters.size(); i++) {
        if (!parameters[i].requires_grad()) continue;
        
        // Initialize moment estimates if needed
        if (m_.size() <= i) {
            m_.push_back(Tensor(parameters[i].shape()));
            v_.push_back(Tensor(parameters[i].shape()));
        }
        
        // Update biased first moment estimate
        m_[i] = beta1_ * m_[i] + (1 - beta1_) * parameters[i].gradient();
        
        // Update biased second raw moment estimate
        v_[i] = beta2_ * v_[i] + (1 - beta2_) * parameters[i].gradient().cwiseProduct(parameters[i].gradient());
        
        // Compute bias-corrected first moment estimate
        Tensor m_hat = m_[i] / (1 - std::pow(beta1_, timestep_));
        
        // Compute bias-corrected second raw moment estimate
        Tensor v_hat = v_[i] / (1 - std::pow(beta2_, timestep_));
        
        // Update parameters
        parameters[i].data() = parameters[i].data() - learning_rate_ * m_hat.cwiseQuotient(v_hat.cwiseSqrt() + epsilon_);
    }
}

} // namespace lm
