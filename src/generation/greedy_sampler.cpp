// src/generation/greedy_sampler.cpp
#include "greedy_sampler.hpp"
#include <algorithm>
#include <stdexcept>

namespace lm {

GreedySampler::GreedySampler() = default;

GreedySampler::~GreedySampler() = default;

int GreedySampler::sample(const Tensor& logits) {
    if (logits.ndim() != 1) {
        throw std::invalid_argument("GreedySampler expects 1D logits tensor");
    }
    
    int vocab_size = logits.dim(0);
    int best_index = 0;
    float best_value = logits(0);
    
    for (int i = 1; i < vocab_size; i++) {
        if (logits(i) > best_value) {
            best_value = logits(i);
            best_index = i;
        }
    }
    
    return best_index;
}

} // namespace lm

