#include "lm/generation/sampler.hpp"
#include <cmath>
#include <queue>
#include <functional>

namespace lm {

int GreedySampler::sample(const Tensor& logits) {
    // Find the token with the highest probability
    const auto& data = logits.data();
    int best_idx = 0;
    float best_val = data(0, 0);
    
    for (int i = 1; i < data.size(); ++i) {
        if (data(i) > best_val) {
            best_val = data(i);
            best_idx = i;
        }
    }
    
    return best_idx;
}

RandomSampler::RandomSampler(float temperature) 
    : temperature_(temperature), gen_(std::random_device{}()) {}

int RandomSampler::sample(const Tensor& logits) {
    // Apply temperature
    Eigen::VectorXf probs = logits.data();
    if (temperature_ != 1.0) {
        probs = probs / temperature_;
    }
    
    // Softmax
    probs = probs.array().exp();
    probs /= probs.sum();
    
    // Sample from distribution
    std::discrete_distribution<int> dist(probs.data(), probs.data() + probs.size());
    return dist(gen_);
}

TopKSampler::TopKSampler(int k, float temperature) 
    : k_(k), temperature_(temperature), gen_(std::random_device{}()) {}

int TopKSampler::sample(const Tensor& logits) {
    // Apply temperature
    Eigen::VectorXf probs = logits.data();
    if (temperature_ != 1.0) {
        probs = probs / temperature_;
    }
    
    // Softmax
    probs = probs.array().exp();
    probs /= probs.sum();
    
    // Create a min-heap to keep track of top-k elements
    using Pair = std::pair<float, int>;
    std::priority_queue<Pair, std::vector<Pair>, std::greater<Pair>> min_heap;
    
    for (int i = 0; i < probs.size(); ++i) {
        min_heap.push({probs(i), i});
        if (min_heap.size() > k_) {
            min_heap.pop();
        }
    }
    
    // Extract indices and probabilities
    std::vector<float> top_probs;
    std::vector<int> top_indices;
    
    while (!min_heap.empty()) {
        top_probs.push_back(min_heap.top().first);
        top_indices.push_back(min_heap.top().second);
        min_heap.pop();
    }
    
    // Normalize
    float sum = std::accumulate(top_probs.begin(), top_probs.end(), 0.0f);
    for (float& p : top_probs) {
        p /= sum;
    }
    
    // Sample from top-k distribution
    std::discrete_distribution<int> dist(top_probs.begin(), top_probs.end());
    return top_indices[dist(gen_)];
}

TopPSampler::TopPSampler(float p, float temperature) 
    : p_(p), temperature_(temperature), gen_(std::random_device{}()) {}

int TopPSampler::sample(const Tensor& logits) {
    // Apply temperature
    Eigen::VectorXf probs = logits.data();
    if (temperature_ != 1.0) {
        probs = probs / temperature_;
    }
    
    // Softmax
    probs = probs.array().exp();
    probs /= probs.sum();
    
    // Create indices and sort by probability
    std::vector<int> indices(probs.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), 
             [&probs](int a, int b) { return probs(a) > probs(b); });
    
    // Find the smallest set of tokens whose cumulative probability >= p
    float cumulative = 0.0f;
    std::vector<float> top_probs;
    std::vector<int> top_indices;
    
    for (int i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        cumulative += probs(idx);
        top_probs.push_back(probs(idx));
        top_indices.push_back(idx);
        
        if (cumulative >= p_) {
            break;
        }
    }
    
    // Renormalize
    for (float& p : top_probs) {
        p /= cumulative;
    }
    
    // Sample from top-p distribution
    std::discrete_distribution<int> dist(top_probs.begin(), top_probs.end());
    return top_indices[dist(gen_)];
}

} // namespace lm
