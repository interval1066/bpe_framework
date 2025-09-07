// src/optimizers/adam.cpp
#include "lm/optimizers/adam.hpp"
#include <fstream>
#include <iostream>
#include <cmath>

namespace lm {

AdamOptimizer::AdamOptimizer(float lr, float b1, float b2, float eps) 
    : learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps), t(0) {}

void AdamOptimizer::initialize_moments(const std::vector<Tensor>& parameters) {
    m.clear();
    v.clear();
    
    for (const auto& param : parameters) {
        // Create zero tensors with the same shape as parameters
        m.push_back(Tensor::zeros(param.shape(), false));
        v.push_back(Tensor::zeros(param.shape(), false));
    }
}

void AdamOptimizer::update(std::vector<Tensor>& parameters, 
                  const std::vector<Tensor>& gradients) {
    // Initialize moments if needed
    if (m.empty() || v.empty()) {
        initialize_moments(parameters);
    }
    
    t++;
    
    for (size_t i = 0; i < parameters.size(); i++) {
        if (!parameters[i].requires_grad()) continue;
        
        // Update biased first moment estimate
        m[i] = m[i] * beta1 + gradients[i] * (1.0f - beta1);
        
        // Update biased second raw moment estimate
        Tensor grad_squared = gradients[i] * gradients[i];
        v[i] = v[i] * beta2 + grad_squared * (1.0f - beta2);
        
        // Compute bias-corrected first moment estimate
        float bias_correction1 = 1.0f - std::pow(beta1, t);
        Tensor m_hat = m[i] / bias_correction1;
        
        // Compute bias-corrected second raw moment estimate
        float bias_correction2 = 1.0f - std::pow(beta2, t);
        Tensor v_hat = v[i] / bias_correction2;
        
        // Update parameters
        Tensor update = m_hat / (v_hat.sqrt() + epsilon);
        parameters[i].data() = parameters[i].data() - learning_rate * update.data();
    }
}

void AdamOptimizer::reset() {
    m.clear();
    v.clear();
    t = 0;
}

void AdamOptimizer::save_state(const std::string& path) const {
    try {
        std::ofstream ofs(path, std::ios::binary);
        cereal::BinaryOutputArchive archive(ofs);
        archive(*this);
    } catch (const std::exception& e) {
        std::cerr << "Error saving AdamOptimizer state: " << e.what() << std::endl;
        throw;
    }
}

void AdamOptimizer::load_state(const std::string& path) {
    try {
        std::ifstream ifs(path, std::ios::binary);
        cereal::BinaryInputArchive archive(ifs);
        archive(*this);
    } catch (const std::exception& e) {
        std::cerr << "Error loading AdamOptimizer state: " << e.what() << std::endl;
        throw;
    }
}

} // namespace lm

