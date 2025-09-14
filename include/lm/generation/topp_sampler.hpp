// include/lm/generation/topp_sampler.hpp
#pragma once

#include "../core/tensor.hpp"
#include "sampler.hpp"
#include <random>
#include <vector>
#include <algorithm>

namespace lm {

class TopPSampler : public Sampler {
public:
    TopPSampler(float p = 0.9f, float temperature = 1.0f);
    ~TopPSampler() override = default;
    
    int sample(const Tensor& logits) override;
    
    void set_p(float p);
    void set_temperature(float temperature);
    float get_p() const;
    float get_temperature() const;
    
private:
    float p_;
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
