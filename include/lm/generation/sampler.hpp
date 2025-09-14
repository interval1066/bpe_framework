#pragma once
#include "lm/core/tensor.hpp"

namespace lm {

class Sampler {
public:
    virtual ~Sampler() = default;
    virtual int sample(const Tensor& logits) = 0;
};

} // namespace lm
