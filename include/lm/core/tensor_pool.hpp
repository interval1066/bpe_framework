#pragma once

#include "tensor.hpp"
#include <vector>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <stdexcept>

namespace lm {

class TensorPool {
private:
    struct TensorKey {
        std::vector<size_t> shape;
        bool requires_grad;

        bool operator==(const TensorKey& other) const {
            return shape == other.shape && requires_grad == other.requires_grad;
        }
    };

    struct KeyHash {
        std::size_t operator()(const TensorKey& k) const {
            std::size_t seed = k.shape.size();
            for (auto& i : k.shape) {
                seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }
            seed ^= k.requires_grad + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            return seed;
        }
    };

    std::unordered_map<TensorKey, std::vector<std::unique_ptr<Tensor>>, KeyHash> pool_;
    mutable std::mutex mutex_;  // Make mutex mutable

public:
    TensorPool() = default;

    std::unique_ptr<Tensor> acquire(const std::vector<size_t>& shape, bool requires_grad = false) {
        TensorKey key{shape, requires_grad};
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = pool_.find(key);
        if (it != pool_.end() && !it->second.empty()) {
            auto tensor = std::move(it->second.back());
            it->second.pop_back();
            return tensor;
        }

        return std::make_unique<Tensor>(shape, requires_grad);
    }

    void release(std::unique_ptr<Tensor> tensor) {
        if (!tensor) return;

        TensorKey key{tensor->shape(), tensor->requires_grad()};
        std::lock_guard<std::mutex> lock(mutex_);

        // Reset tensor state before pooling
        tensor->zero_grad();
        tensor->data().setZero();
        
        pool_[key].push_back(std::move(tensor));
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        pool_.clear();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        size_t total = 0;
        for (const auto& entry : pool_) {
            total += entry.second.size();
        }
        return total;
    }
};

} // namespace lm
