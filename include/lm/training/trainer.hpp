// include/lm/training/trainer.hpp
#pragma once

#include <string>
#include "../models/language_model.hpp"
#include "../optimizers/adam.hpp"

namespace lm {
namespace training {

struct TrainingCheckpoint {
    size_t epoch;
    size_t iteration;
    float loss;
    
    template <class Archive>
    void serialize(Archive& archive) {
        archive(epoch, iteration, loss);
    }
};

class Trainer {
private:
    LanguageModel& model;
    AdamOptimizer& optimizer;
    
public:
    Trainer(LanguageModel& model, AdamOptimizer& optimizer);
    
    void train(const std::vector<std::string>& corpus, 
               size_t num_epochs, 
               size_t batch_size, 
               size_t sequence_length);
    
    void save_checkpoint(const std::string& path, 
                        const TrainingCheckpoint& checkpoint) const;
    TrainingCheckpoint load_checkpoint(const std::string& path);
};

} // namespace training
} // namespace lm

