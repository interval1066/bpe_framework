#pragma once

#include "../core/tensor.hpp"
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>

namespace lm {

class Sampler {
public:
    virtual ~Sampler() = default;
    virtual int sample(const Tensor& logits) = 0;
};

class GreedySampler : public Sampler {
public:
    int sample(const Tensor& logits) override;
};

class RandomSampler : public Sampler {
public:
    RandomSampler(float temperature = 1.0);
    int sample(const Tensor& logits) override;
    
private:
    float temperature_;
    std::mt19937 gen_;
};

class TopKSampler : public Sampler {
public:
    TopKSampler(int k, float temperature = 1.0);
    int sample(const Tensor& logits) override;
    
private:
    int k_;
    float temperature_;
    std::mt19937 gen_;
};

class TopPSampler : public Sampler {
public:
    TopPSampler(float p, float temperature = 1.0);
    int sample(const Tensor& logits) override;
    
private:
    float p_;
    float temperature_;
    std::mt19937 gen_;
};

} // namespace lm
