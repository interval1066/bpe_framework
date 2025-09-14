#include "random_sampler.hpp"
#include <cmath>
#include <stdexcept>

namespace lm {

RandomSampler::RandomSampler(float temperature) 
    : temperature_(temperature), rng_(std::random_device{}()), dist_(0.0f, 1.0f) {
    if (temperature <= 0.0f) {
        throw std::invalid_argument("Temperature must be positive");
    }
}

int RandomSampler::sample(const Tensor& logits) {
    if (logits.ndim() != 1) {
        throw std::invalid_argument("RandomSampler expects 1D logits tensor");
    }
    
    int vocab_size = logits.dim(0);
    
    // Apply temperature scaling
    std::vector<float> scaled_logits(vocab_size);
    for (int i = 0; i < vocab_size; i++) {
        scaled_logits[i] = logits(i) / temperature_;
    }
    
    // Convert scaled logits to probabilities using softmax
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
    std::vector<float> probs(vocab_size);
    for (int i = 0; i < vocab_size; i++) {
        probs[i] = scaled_logits[i] / sum_exp;
    }
    
    // Sample from the distribution
    float random_value = dist_(rng_);
    float cumulative_prob = 0.0f;
    
    for (int i = 0; i < vocab_size; i++) {
        cumulative_prob += probs[i];
        if (random_value <= cumulative_prob) {
            return i;
        }
    }
    
    // Fallback: return the last token
    return vocab_size - 1;
}

void RandomSampler::set_temperature(float temperature) {
    if (temperature <= 0.0f) {
        throw std::invalid_argument("Temperature must be positive");
    }
    temperature_ = temperature;
}

float RandomSampler::get_temperature() const {
    return temperature_;
}

} // namespace lm

