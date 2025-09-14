// include/lm/generation/topk_sampler.hpp
#pragma once

#include "../core/tensor.hpp"
#include "sampler.hpp"
#include <random>
#include <vector>
#include <algorithm>

namespace lm {

class TopKSampler : public Sampler {
public:
    TopKSampler(int k = 50, float temperature = 1.0f);
    ~TopKSampler() override = default;
    
    int sample(const Tensor& logits) override;
    
    void set_k(int k);
    void set_temperature(float temperature);
    int get_k() const;
    float get_temperature() const;
    
private:
    int k_;
    float temperature_;
    std::mt19937 rng_;
    
    struct TokenProbability {
        int token_id;
        float probability;
        
        bool operator<(const TokenProbability& other) const {
            return probability > other.probability; // Sort descending
        }
    };
};

} // namespace lm
