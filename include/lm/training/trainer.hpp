#pragma once

#include "lm/models/language_model.hpp"
#include "lm/tokenizer/bpe_tokenizer.hpp"
#include "lm/optimizers/adam.hpp"
#include <vector>
#include <string>

namespace lm {

class LanguageModelTrainer {
public:
    // Change to accept a reference
    LanguageModelTrainer(const BPETokenizer& tokenizer,
                       size_t embedding_dim,
                       size_t hidden_dim,
                       size_t num_layers);
    
    void train(const std::vector<std::string>& corpus, 
              size_t epochs, 
              size_t batch_size, 
              size_t sequence_length);
    
    Tensor prepare_batch(const std::vector<std::string>& texts, 
                       size_t sequence_length);
    
    float compute_loss(const Tensor& logits, const Tensor& targets);
    
    void save_model(const std::string& path);
    void load_model(const std::string& path);

private:
    const BPETokenizer& tokenizer_;  // Store a reference instead of a copy
    LanguageModel model_;
    AdamOptimizer optimizer_;
};

} // namespace lm
