// src/generation/topk_sampler.cpp
// Limits the sampling to the top K most probable tokens
#include "topk_sampler.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace lm {

TopKSampler::TopKSampler(int k, float temperature) 
    : k_(k), temperature_(temperature), rng_(std::random_device{}()) {
    if (k <= 0) {
        throw std::invalid_argument("K must be positive");
    }
    if (temperature <= 0.0f) {
        throw std::invalid_argument("Temperature must be positive");
    }
}

int TopKSampler::sample(const Tensor& logits) {
    if (logits.ndim() != 1) {
        throw std::invalid_argument("TopKSampler expects 1D logits tensor");
    }
    
    int vocab_size = logits.dim(0);
    k_ = std::min(k_, vocab_size); // Ensure k doesn't exceed vocabulary size
    
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
    
    // Take top k tokens
    std::vector<TokenProbability> top_k(token_probs.begin(), token_probs.begin() + k_);
    
    // Renormalize probabilities
    float top_k_sum = 0.0f;
    for (auto& tp : top_k) {
        top_k_sum += tp.probability;
    }
    for (auto& tp : top_k) {
        tp.probability /= top_k_sum;
    }
    
    // Sample from the top-k distribution
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float random_value = dist(rng_);
    float cumulative_prob = 0.0f;
    
    for (const auto& tp : top_k) {
        cumulative_prob += tp.probability;
        if (random_value <= cumulative_prob) {
            return tp.token_id;
        }
    }
    
    // Fallback: return the most probable token
    return top_k[0].token_id;
}

void TopKSampler::set_k(int k) {
    if (k <= 0) {
        throw std::invalid_argument("K must be positive");
    }
    k_ = k;
}

void TopKSampler::set_temperature(float temperature) {
    if (temperature <= 0.0f) {
        throw std::invalid_argument("Temperature must be positive");
    }
    temperature_ = temperature;
}

int TopKSampler::get_k() const {
    return k_;
}

float TopKSampler::get_temperature() const {
    return temperature_;
}

} // namespace lm
