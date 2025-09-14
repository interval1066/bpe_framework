// include/lm/generation/temperature_sampler.hpp
#pragma once

#include "../core/tensor.hpp"
#include "sampler.hpp"
#include <random>

namespace lm {

class TemperatureSampler : public Sampler {
public:
    TemperatureSampler(float temperature = 1.0f);
    ~TemperatureSampler() override = default;
    
    int sample(const Tensor& logits) override;
    
    void set_temperature(float temperature);
    float get_temperature() const;
    
private:
    float temperature_;
    std::mt19937 rng_;
};

} // namespace lm
