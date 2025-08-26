#include "lm/models/feed_forward.hpp"
#include <cmath>
#include <iostream>
#include <random>

namespace lm {

FeedForward::FeedForward(size_t d_model, size_t d_ff, float dropout)
    : d_model_(d_model), d_ff_(d_ff), dropout_(dropout) {
    
    // Initialize weight matrices and biases
    w1_ = Tensor::xavier(std::vector<size_t>{d_model_, d_ff_});
    b1_ = Tensor::zeros(std::vector<size_t>{d_ff_});
    w2_ = Tensor::xavier(std::vector<size_t>{d_ff_, d_model_});
    b2_ = Tensor::zeros(std::vector<size_t>{d_model_});
    
    std::cout << "Initialized FeedForward with:\n";
    std::cout << "  d_model: " << d_model_ << "\n";
    std::cout << "  d_ff: " << d_ff_ << "\n";
    std::cout << "  dropout: " << dropout_ << "\n";
}

std::vector<Tensor> FeedForward::parameters() const {
    return {w1_, b1_, w2_, b2_};
}

void FeedForward::set_training(bool training) {
    training_ = training;
}

Tensor FeedForward::forward(const Tensor& input) const {
    // Get input dimensions
    size_t batch_size = input.shape()[0];
    size_t seq_len = input.shape()[1];
    
    // First linear transformation: input * w1 + b1
    Tensor hidden(std::vector<size_t>{batch_size, seq_len, d_ff_});
    
    // Calculate strides for flat indexing
    size_t input_stride_1 = d_model_;  // stride for sequence position in input
    size_t hidden_stride_1 = d_ff_;    // stride for sequence position in hidden
    
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t t = 0; t < seq_len; ++t) {
            for (size_t f = 0; f < d_ff_; ++f) {
                // Calculate flat index for hidden
                size_t hidden_index = b * seq_len * hidden_stride_1 + 
                                     t * hidden_stride_1 + 
                                     f;
                
                // Initialize with bias
                hidden(hidden_index) = b1_(f);
                
                for (size_t d = 0; d < d_model_; ++d) {
                    // Calculate flat index for input
                    size_t input_index = b * seq_len * input_stride_1 + 
                                       t * input_stride_1 + 
                                       d;
                    
                    hidden(hidden_index) += input(input_index) * w1_(d, f);
                }
            }
        }
    }
    
    // GELU activation
    hidden = gelu(hidden);
    
    // Apply dropout during training
    if (training_) {
        hidden = apply_dropout(hidden, dropout_);
    }
    
    // Second linear transformation: hidden * w2 + b2
    Tensor output(std::vector<size_t>{batch_size, seq_len, d_model_});
    
    // Calculate strides for output
    size_t output_stride_1 = d_model_;  // stride for sequence position in output
    
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t t = 0; t < seq_len; ++t) {
            for (size_t d = 0; d < d_model_; ++d) {
                // Calculate flat index for output
                size_t output_index = b * seq_len * output_stride_1 + 
                                    t * output_stride_1 + 
                                    d;
                
                // Initialize with bias
                output(output_index) = b2_(d);
                
                for (size_t f = 0; f < d_ff_; ++f) {
                    // Calculate flat index for hidden
                    size_t hidden_index = b * seq_len * hidden_stride_1 + 
                                        t * hidden_stride_1 + 
                                        f;
                    
                    output(output_index) += hidden(hidden_index) * w2_(f, d);
                }
            }
        }
    }
    
    return output;
}

Tensor FeedForward::gelu(const Tensor& input) const {
    // GELU activation function: x * 0.5 * (1.0 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float sqrt_2_over_pi = std::sqrt(2.0f / M_PI);
    Tensor result(input.shape());
    
    for (size_t i = 0; i < input.size(); ++i) {
        float x = input(i);
        float x_cubed = x * x * x;
        result(i) = 0.5f * x * (1.0f + std::tanh(sqrt_2_over_pi * (x + 0.044715f * x_cubed)));
    }
    
    return result;
}

Tensor FeedForward::apply_dropout(const Tensor& input, float dropout_rate) const {
    if (dropout_rate <= 0.0) return input;
    
    Tensor output = input;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::bernoulli_distribution dist(1.0 - dropout_rate);
    
    for (size_t i = 0; i < output.size(); ++i) {
        if (!dist(gen)) {
            output(i) = 0.0;
        } else {
            output(i) /= (1.0 - dropout_rate);
        }
    }
    
    return output;
}

} // namespace lm
