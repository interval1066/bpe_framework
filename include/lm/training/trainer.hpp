#pragma once

#include "lm/models/language_model.hpp"
#include "lm/tokenizer/bpe_tokenizer.hpp"
#include "lm/optimizers/adam.hpp"
#include <vector>
#include <string>

namespace lm {

class LanguageModelTrainer {
public:
    LanguageModelTrainer(const BPETokenizer& tokenizer,
                       size_t embedding_dim,
                       size_t hidden_dim,
                       size_t num_layers);
    
    void train(const std::vector<std::string>& corpus, 
              size_t epochs, 
              size_t batch_size, 
              size_t sequence_length);
    
    void save_model(const std::string& path);
    void load_model(const std::string& path);
    
    // Model accessor methods
    LanguageModel& model() { return model_; }
    const LanguageModel& model() const { return model_; }
    
    Tensor prepare_batch(const std::vector<std::string>& texts, 
                       size_t sequence_length);
    
    float compute_loss(const Tensor& logits, const Tensor& targets);

private:
    const BPETokenizer& tokenizer_;
    LanguageModel model_;
    AdamOptimizer optimizer_;
};

} // namespace lm

