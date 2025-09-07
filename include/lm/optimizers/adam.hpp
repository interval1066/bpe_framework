// include/lm/optimizers/adam.hpp
#pragma once

#include <vector>
#include <cmath>
#include <cereal/types/vector.hpp>
#include <cereal/archives/binary.hpp>
#include "../core/tensor.hpp"

namespace lm {

class AdamOptimizer {
private:
    std::vector<Tensor> m;  // First moment vector
    std::vector<Tensor> v;  // Second moment vector
    size_t t;               // Timestep
    float beta1;
    float beta2;
    float epsilon;
    float learning_rate;

public:
    AdamOptimizer(float lr = 0.001, float b1 = 0.9, float b2 = 0.999, float eps = 1e-8);
    
    void update(std::vector<Tensor>& parameters, 
                const std::vector<Tensor>& gradients);
    
    // Initialize moment vectors for parameters
    void initialize_moments(const std::vector<Tensor>& parameters);
    
    // Reset the optimizer state
    void reset();
    
    // Step function for compatibility with existing code
    void step(std::vector<Tensor>& parameters) {
        std::vector<Tensor> gradients;
        for (auto& param : parameters) {
            if (param.requires_grad()) {
                gradients.push_back(param.grad());
            } else {
                gradients.push_back(Tensor::zeros(param.shape(), false));
            }
        }
        update(parameters, gradients);
    }
    
    void zero_grad(std::vector<Tensor>& parameters) {
        for (auto& param : parameters) {
            if (param.requires_grad()) {
                param.zero_grad();
            }
        }
    }
    
    // Serialization methods
    void save_state(const std::string& path) const;
    void load_state(const std::string& path);
    
    // Cereal serialization
    template <class Archive>
    void serialize(Archive& archive) {
        archive(
            cereal::make_nvp("m", m),
            cereal::make_nvp("v", v),
            cereal::make_nvp("t", t),
            cereal::make_nvp("beta1", beta1),
            cereal::make_nvp("beta2", beta2),
            cereal::make_nvp("epsilon", epsilon),
            cereal::make_nvp("learning_rate", learning_rate)
        );
    }
    
    // Getters for state inspection
    size_t get_timestep() const { return t; }
    float get_learning_rate() const { return learning_rate; }
    void set_learning_rate(float lr) { learning_rate = lr; }
};

} // namespace lm

