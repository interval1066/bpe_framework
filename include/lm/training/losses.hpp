// include/lm/training/losses.hpp
#pragma once

#include "../core/tensor.hpp"

namespace lm {

Tensor cross_entropy_loss(const Tensor& logits, const Tensor& targets, const Tensor& mask = Tensor());

} // namespace lm

