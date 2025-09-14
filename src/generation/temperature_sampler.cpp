// src/generation/temperature_sampler.cpp
#include "temperature_sampler.hpp"
#include <cmath>
#include <random>
#include <stdexcept>

namespace lm {

TemperatureSampler::TemperatureSampler(float temperature) 
    : temperature_(temperature), rng_(std::random_device{}()) {
    if (temperature <= 0.0f) {
        throw std::invalid_argument("Temperature must be positive");
    }
}

int TemperatureSampler::sample(const Tensor& logits) {
    if (logits.ndim() != 1) {
        throw std::invalid_argument("TemperatureSampler expects 1D logits tensor");
    }
    
    int vocab_size = logits.dim(0);
    
    // Apply temperature scaling
    std::vector<float> scaled_logits(vocab_size);
    for (int i = 0; i < vocab_size; i++) {
        scaled_logits[i] = logits(i) / temperature_;
    }
    
    // Compute softmax
    float max_logit = scaled_logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (scaled_logits[i] > max_logit) {
            max_logit = scaled_logits[i];
        }
    }
    
    float sum_exp = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        scaled_logits[i] = std::exp(scaled_logits[i] - max_logit);
        sum_exp += scaled_logits[i];
    }
    
    // Normalize to get probabilities
    for (int i = 0; i < vocab_size; i++) {
        scaled_logits[i] /= sum_exp;
    }
    
    // Sample from the distribution
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float random_value = dist(rng_);
    float cumulative_prob = 0.0f;
    
    for (int i = 0; i < vocab_size; i++) {
        cumulative_prob += scaled_logits[i];
        if (random_value <= cumulative_prob) {
            return i;
        }
    }
    
    // Fallback: return the most probable token
    return 0;
}

void TemperatureSampler::set_temperature(float temperature) {
    if (temperature <= 0.0f) {
        throw std::invalid_argument("Temperature must be positive");
    }
    temperature_ = temperature;
}

float TemperatureSampler::get_temperature() const {
    return temperature_;
}

} // namespace lm

