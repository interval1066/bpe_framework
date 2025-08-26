#include "lm/optimizers/adam.hpp"
#include <cmath>

namespace lm {

AdamOptimizer::AdamOptimizer(float learning_rate, float beta1, float beta2, float epsilon)
    : learning_rate_(learning_rate), beta1_(beta1), beta2_(beta2), epsilon_(epsilon), timestep_(0) {}

void AdamOptimizer::zero_grad(std::vector<Tensor>& parameters) {
    for (auto& param : parameters) {
        if (param.requires_grad()) {
            param.zero_grad();
        }
    }
}

void AdamOptimizer::step(std::vector<Tensor>& parameters) {
    timestep_++;
    
    for (size_t i = 0; i < parameters.size(); i++) {
        if (!parameters[i].requires_grad()) continue;
        
        // Initialize moment estimates if needed
        if (m_.size() <= i) {
            m_.push_back(Tensor::zeros(parameters[i].shape()));
            v_.push_back(Tensor::zeros(parameters[i].shape()));
        }
        
        // Convert gradient to Tensor for consistent operations
        Tensor grad_tensor(parameters[i].grad(), parameters[i].shape());
        
        // Update biased first moment estimate using Tensor operations
        m_[i] = m_[i] * beta1_ + grad_tensor * (1 - beta1_);
        
        // Update biased second raw moment estimate using Tensor operations
        Tensor grad_squared = grad_tensor * grad_tensor;
        v_[i] = v_[i] * beta2_ + grad_squared * (1 - beta2_);
        
        // Compute bias-corrected first moment estimate
        float bias_correction1 = 1 - std::pow(beta1_, timestep_);
        Tensor m_hat = m_[i] / bias_correction1;
        
        // Compute bias-corrected second raw moment estimate
        float bias_correction2 = 1 - std::pow(beta2_, timestep_);
        Tensor v_hat = v_[i] / bias_correction2;
        
        // Update parameters using Tensor operations
        Tensor update = m_hat / (v_hat.sqrt() +
            Tensor(Eigen::MatrixXf::Constant(v_hat.data().rows(), v_hat.data().cols(), epsilon_),
            v_hat.shape()));
        parameters[i].data() -= learning_rate_ * update.data();
    }
}

} // namespace lm

