// src/training/trainer.cpp
#include "lm/training/trainer.hpp"
#include <fstream>

namespace lm {
namespace training {

Trainer::Trainer(LanguageModel& model, AdamOptimizer& optimizer)  // Remove models:: prefix
    : model(model), optimizer(optimizer) {}

void Trainer::save_checkpoint(const std::string& path, 
                             const TrainingCheckpoint& checkpoint) const {
    std::ofstream ofs(path, std::ios::binary);
    cereal::BinaryOutputArchive archive(ofs);
    
    // Save training state
    archive(checkpoint);
    
    // Save model parameters
    auto params = model.get_parameters();
    archive(params);
    
    // Save optimizer state to a separate file
    optimizer.save_state(path + ".optim");
}

TrainingCheckpoint Trainer::load_checkpoint(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    cereal::BinaryInputArchive archive(ifs);
    
    TrainingCheckpoint checkpoint;
    archive(checkpoint);
    
    // Load model parameters
    std::vector<Tensor> params;
    archive(params);
    model.set_parameters(params);
    
    // Load optimizer state
    optimizer.load_state(path + ".optim");
    
    return checkpoint;
}

void Trainer::train(size_t num_epochs) {
    TrainingCheckpoint checkpoint {0, 0, 0.0f};
    
    // Try to resume from checkpoint
    try {
        checkpoint = load_checkpoint("checkpoint.bin");
        std::cout << "Resumed from epoch " << checkpoint.epoch << std::endl;
    } catch (...) {
        std::cout << "Starting new training" << std::endl;
    }
    
    for (size_t epoch = checkpoint.epoch; epoch < num_epochs; ++epoch) {
        // Training logic...
        
        // Save checkpoint periodically
        if (epoch % 10 == 0) {
            checkpoint.epoch = epoch;
            save_checkpoint("checkpoint.bin", checkpoint);
        }
    }
}

} // namespace training
} // namespace lm

