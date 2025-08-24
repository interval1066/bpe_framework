#include "lm/models/feed_forward.hpp"
#include <cmath>
#include <iostream>
#include <random>

namespace lm {

FeedForward::FeedForward(size_t d_model, size_t d_ff, float dropout)
    : d_model_(d_model), d_ff_(d_ff), dropout_(dropout) {
    
    // Initialize weight matrices and biases
    w1_ = Tensor::xavier({d_model_, d_ff_});
    b1_ = Tensor::zeros({d_ff_});
    w2_ = Tensor::xavier({d_ff_, d_model_});
    b2_ = Tensor::zeros({d_model_});
    
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

Tensor FeedForward::forward(const Tensor& input) {
    // Get input dimensions
    size_t batch_size = input.shape()[0];
    size_t seq_len = input.shape()[1];
    
    // First linear transformation: input * w1 + b1
    Tensor hidden({batch_size, seq_len, d_ff_});
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t t = 0; t < seq_len; ++t) {
            for (size_t f = 0; f < d_ff_; ++f) {
                hidden(b, t, f) = b1_(f);
                for (size_t d = 0; d < d_model_; ++d) {
                    hidden(b, t, f) += input(b, t, d) * w1_(d, f);
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
    Tensor output({batch_size, seq_len, d_model_});
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t t = 0; t < seq_len; ++t) {
            for (size_t d = 0; d < d_model_; ++d) {
                output(b, t, d) = b2_(d);
                for (size_t f = 0; f < d_ff_; ++f) {
                    output(b, t, d) += hidden(b, t, f) * w2_(f, d);
                }
            }
        }
    }
    
    return output;
}

Tensor FeedForward::gelu(const Tensor& input) {
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

Tensor FeedForward::apply_dropout(const Tensor& input, float dropout_rate) {
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

