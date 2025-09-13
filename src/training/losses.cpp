// src/training/losses.cpp
#include "losses.hpp"
#include <cmath>
#include <stdexcept>

namespace lm {

Tensor cross_entropy_loss(const Tensor& logits, const Tensor& targets, const Tensor& mask) {
    if (logits.shape().size() != 3) {
        throw std::invalid_argument("Logits must be 3D tensor [batch, seq_len, vocab_size]");
    }
    
    if (targets.shape().size() != 2) {
        throw std::invalid_argument("Targets must be 2D tensor [batch, seq_len]");
    }
    
    size_t batch_size = logits.shape()[0];
    size_t seq_len = logits.shape()[1];
    size_t vocab_size = logits.shape()[2];
    
    if (targets.shape()[0] != batch_size || targets.shape()[1] != seq_len) {
        throw std::invalid_argument("Logits and targets must have compatible shapes");
    }
    
    // Create output tensor
    Tensor loss({batch_size, seq_len}, false);
    
    // Compute cross-entropy loss
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t s = 0; s < seq_len; s++) {
            int target_idx = static_cast<int>(targets(b, s));
            
            // Skip padded positions (target = -100)
            if (target_idx == -100) {
                loss(b, s) = 0.0f;
                continue;
            }
            
            if (target_idx < 0 || target_idx >= static_cast<int>(vocab_size)) {
                throw std::out_of_range("Target index out of vocabulary range");
            }
            
            // Compute softmax and cross-entropy for this position
            float max_logit = logits(b, s, 0);
            for (size_t v = 1; v < vocab_size; v++) {
                if (logits(b, s, v) > max_logit) {
                    max_logit = logits(b, s, v);
                }
            }
            
            float sum_exp = 0.0f;
            for (size_t v = 0; v < vocab_size; v++) {
                sum_exp += std::exp(logits(b, s, v) - max_logit);
            }
            
            float log_softmax = logits(b, s, target_idx) - max_logit - std::log(sum_exp);
            loss(b, s) = -log_softmax;
        }
    }
    
    // If mask is provided, apply it
    if (mask.shape().size() > 0) {
        if (mask.shape()[0] != batch_size || mask.shape()[1] != seq_len) {
            throw std::invalid_argument("Mask must have same shape as loss");
        }
        
        for (size_t b = 0; b < batch_size; b++) {
            for (size_t s = 0; s < seq_len; s++) {
                loss(b, s) *= mask(b, s);
            }
        }
    }
    
    return loss;
}

} // namespace lm

