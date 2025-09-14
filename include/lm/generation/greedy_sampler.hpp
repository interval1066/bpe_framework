// include/lm/generation/greedy_sampler.hpp
#pragma once

#include "lm/core/tensor.hpp"
#include "lm/generation/sampler.hpp"
#include <stdexcept>

namespace lm {

class GreedySampler : public Sampler {
public:
    GreedySampler();
    ~GreedySampler() override;
    
    int sample(const Tensor& logits) override;
};

} // namespace lm

