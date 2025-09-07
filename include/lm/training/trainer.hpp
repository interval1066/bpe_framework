// include/lm/training/trainer.hpp
#pragma once

#include <string>
#include <cereal/types/vector.hpp>
#include <cereal/archives/binary.hpp>
#include "../optimizers/adam.hpp"
#include "../models/language_model.hpp"  // Add this include

namespace lm {
namespace training {

struct TrainingCheckpoint {
    size_t epoch;
    size_t iteration;
    float loss;
    
    // Cereal serialization
    template <class Archive>
    void serialize(Archive& archive) {
        archive(
            cereal::make_nvp("epoch", epoch),
            cereal::make_nvp("iteration", iteration),
            cereal::make_nvp("loss", loss)
        );
    }
};

class Trainer {
private:
    LanguageModel& model;  // Remove models:: prefix
    AdamOptimizer& optimizer;
    // ... other members
    
public:
    Trainer(LanguageModel& model, AdamOptimizer& optimizer);  // Remove models:: prefix
    
    void train(size_t num_epochs);
    
    void save_checkpoint(const std::string& path, 
                        const TrainingCheckpoint& checkpoint) const;
    TrainingCheckpoint load_checkpoint(const std::string& path);
};

} // namespace training
} // namespace lm
