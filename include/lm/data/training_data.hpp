// include/lm/data/training_data.hpp
#pragma once

#include <vector>
#include <string>
#include <utility>
#include <random>
#include <algorithm>

namespace lm {

struct TrainingExample {
    std::string input;
    std::string target;
    
    TrainingExample(const std::string& i, const std::string& t) 
        : input(i), target(t) {}
};

class TrainingDataset {
private:
    std::vector<TrainingExample> examples_;
    
public:
    void add_example(const std::string& input, const std::string& target) {
        examples_.emplace_back(input, target);
    }
    
    void add_conversation(const std::vector<std::string>& turns) {
        for (size_t i = 0; i < turns.size() - 1; i++) {
            examples_.emplace_back(turns[i], turns[i + 1]);
        }
    }
    
    const std::vector<TrainingExample>& examples() const { return examples_; }
    size_t size() const { return examples_.size(); }
    
    void shuffle() {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(examples_.begin(), examples_.end(), g);
    }
    
    void clear() { examples_.clear(); }
};

} // namespace lm
