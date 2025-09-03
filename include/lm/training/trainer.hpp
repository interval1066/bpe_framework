#pragma once

#include "lm/models/language_model.hpp"
#include "lm/tokenizer/bpe_tokenizer.hpp"
#include "lm/optimizers/adam.hpp"
#include <vector>
#include <string>
#include <memory>
#include <functional>

namespace lm {

class Sampler;

class LanguageModelTrainer {
public:
    // Callback type for training progress monitoring
    using ProgressCallback = std::function<void(
        size_t epoch, 
        float train_loss, 
        float val_loss, 
        float learning_rate
    )>;
    
    LanguageModelTrainer(const BPETokenizer& tokenizer,
                       size_t embedding_dim,
                       size_t hidden_dim,
                       size_t num_layers);
    
    // Main training method with validation
    void train(const std::vector<std::string>& corpus, 
              size_t epochs, 
              size_t batch_size, 
              size_t sequence_length,
              float validation_split = 0.15f,
              size_t validation_freq = 1,
              size_t early_stopping_patience = 0,
              const ProgressCallback& callback = nullptr);
    
    // Evaluate on a separate dataset
    float evaluate(const std::vector<std::string>& corpus,
                  size_t batch_size,
                  size_t sequence_length);
    
    // Test the model on held-out data
    std::pair<float, float> test(const std::vector<std::string>& corpus,
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
    
    // Get training history
    const std::vector<float>& get_train_loss_history() const { return train_loss_history_; }
    const std::vector<float>& get_val_loss_history() const { return val_loss_history_; }

    std::string generate(const std::string& prompt, 
        size_t max_length, Sampler& sampler,
        size_t sequence_length);
    
    std::vector<int> generate_tokens(const std::string& prompt, 
       size_t max_length, Sampler& sampler, size_t sequence_length);
    
    // Batch generation
    std::vector<std::string> generate_batch(const std::vector<std::string>& prompts, 
                                           size_t max_length, 
                                           Sampler& sampler,
                                           size_t sequence_length,
                                           size_t batch_size = 1);
 
private:
    const BPETokenizer& tokenizer_;
    LanguageModel model_;
    AdamOptimizer optimizer_;
    
    // Training history
    std::vector<float> train_loss_history_;
    std::vector<float> val_loss_history_;
    
    // Helper methods
    std::pair<std::vector<std::string>, std::vector<std::string>> 
    split_data(const std::vector<std::string>& corpus, float validation_split);
    
    float run_validation(const std::vector<std::string>& validation_data,
        size_t batch_size, size_t sequence_length);

    Tensor prepare_inference_batch(const std::vector<std::vector<int>>& token_sequences, 
        size_t sequence_length);
};

} // namespace lm

