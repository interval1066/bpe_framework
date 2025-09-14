// src/generation/topp_sampler.cpp
// Limits the sampling to the smallest set of tokens whose 
#include "topp_sampler.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <numeric>

namespace lm {

TopPSampler::TopPSampler(float p, float temperature) 
    : p_(p), temperature_(temperature), rng_(std::random_device{}()) {
    if (p <= 0.0f || p > 1.0f) {
        throw std::invalid_argument("P must be in range (0, 1]");
    }
    if (temperature <= 0.0f) {
        throw std::invalid_argument("Temperature must be positive");
    }
}

int TopPSampler::sample(const Tensor& logits) {
    if (logits.ndim() != 1) {
        throw std::invalid_argument("TopPSampler expects 1D logits tensor");
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
    
    // Create and sort token-probability pairs
    std::vector<TokenProbability> token_probs;
    for (int i = 0; i < vocab_size; i++) {
        token_probs.push_back({i, scaled_logits[i] / sum_exp});
    }
    
    // Sort by probability (descending)
    std::sort(token_probs.begin(), token_probs.end());
    
    // Find the smallest set of tokens whose cumulative probability >= p
    float cumulative_prob = 0.0f;
    size_t nucleus_size = 0;
    
    for (size_t i = 0; i < token_probs.size(); i++) {
        cumulative_prob += token_probs[i].probability;
        if (cumulative_prob >= p_) {
            nucleus_size = i + 1;
            break;
        }
    }
    
    // If we didn't reach p, use all tokens
    if (nucleus_size == 0) {
        nucleus_size = token_probs.size();
    }
    
    // Take nucleus tokens
    std::vector<TokenProbability> nucleus(token_probs.begin(), token_probs.begin() + nucleus_size);
    
    // Renormalize probabilities
    float nucleus_sum = 0.0f;
    for (auto& tp : nucleus) {
        nucleus_sum += tp.probability;
    }
    for (auto& tp : nucleus) {
        tp.probability /= nucleus_sum;
    }
    
    // Sample from the nucleus distribution
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float random_value = dist(rng_);
    cumulative_prob = 0.0f;
    
    for (const auto& tp : nucleus) {
        cumulative_prob += tp.probability;
        if (random_value <= cumulative_prob) {
            return tp.token_id;
        }
    }
    
    // Fallback: return the most probable token
    return nucleus[0].token_id;
}

void TopPSampler::set_p(float p) {
    if (p <= 0.0f || p > 1.0f) {
        throw std::invalid_argument("P must be in range (0, 1]");
    }
    p_ = p;
}

void TopPSampler::set_temperature(float temperature) {
    if (temperature <= 0.0f) {
        throw std::invalid_argument("Temperature must be positive");
    }
    temperature_ = temperature;
}

float TopPSampler::get_p() const {
    return p_;
}

float TopPSampler::get_temperature() const {
    return temperature_;
}

} // namespace lm
