#pragma once

#include "../core/tensor.hpp"
#include "sampler.hpp"
#include <random>

namespace lm {

class RandomSampler : public Sampler {
public:
    RandomSampler(float temperature = 1.0f);
    ~RandomSampler() override = default;
    
    int sample(const Tensor& logits) override;
    
    void set_temperature(float temperature);
    float get_temperature() const;
    
private:
    float temperature_;
    std::mt19937 rng_;
    std::uniform_real_distribution<float> dist_;
};

} // namespace lm

