#include "lm/models/layer_norm.hpp"
#include <cmath>
#include <iostream>

namespace lm {

LayerNorm::LayerNorm(size_t d_model, float eps)
    : d_model_(d_model), eps_(eps) {
    
    // Initialize gamma (scale) to ones and beta (bias) to zeros
    gamma_ = Tensor::ones(std::vector<size_t>{d_model_});
    beta_ = Tensor::zeros(std::vector<size_t>{d_model_});
    
    std::cout << "Initialized LayerNorm with:\n";
    std::cout << "  d_model: " << d_model_ << "\n";
    std::cout << "  eps: " << eps_ << "\n";
}

std::vector<Tensor> LayerNorm::parameters() const {
    return {gamma_, beta_};
}

void LayerNorm::set_training(/*bool training*/) {
    // LayerNorm doesn't have different behavior during training vs evaluation
    // This method is here for interface consistency
}

Tensor LayerNorm::forward(const Tensor& input) {
    // Get input dimensions
    size_t batch_size = input.shape()[0];
    size_t seq_len = input.shape()[1];
    
    // Create output tensor with same shape as input
    Tensor output(input.shape());
    
    // Calculate strides for flat indexing
    size_t input_stride_1 = d_model_;  // stride for sequence position in input
    size_t input_stride_2 = 1;         // stride for feature dimension in input
    
    // For each element in the batch and each position in the sequence
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t t = 0; t < seq_len; ++t) {
            // Calculate mean
            float mean = 0.0f;
            for (size_t d = 0; d < d_model_; ++d) {
                size_t input_index = b * seq_len * input_stride_1 + 
                                   t * input_stride_1 + 
                                   d * input_stride_2;
                mean += input(input_index);
            }
            mean /= d_model_;
            
            // Calculate variance
            float variance = 0.0f;
            for (size_t d = 0; d < d_model_; ++d) {
                size_t input_index = b * seq_len * input_stride_1 + 
                                   t * input_stride_1 + 
                                   d * input_stride_2;
                float diff = input(input_index) - mean;
                variance += diff * diff;
            }
            variance /= d_model_;
            
            // Normalize
            for (size_t d = 0; d < d_model_; ++d) {
                size_t input_index = b * seq_len * input_stride_1 + 
                                   t * input_stride_1 + 
                                   d * input_stride_2;
                size_t output_index = b * seq_len * input_stride_1 + 
                                    t * input_stride_1 + 
                                    d * input_stride_2;
                
                float normalized = (input(input_index) - mean) / std::sqrt(variance + eps_);
                output(output_index) = gamma_(d) * normalized + beta_(d);
            }
        }
    }
    
    return output;
}

} // namespace lm

