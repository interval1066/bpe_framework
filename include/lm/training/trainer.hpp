#pragma once

#include <vector>
#include <string>
#include <functional>
#include "lm/models/language_model.hpp"
#include "lm/tokenizer/bpe_tokenizer.hpp"
#include "lm/optimizers/adam.hpp"
#include "lm/conversation.hpp"
#include "lm/generation/sampler.hpp"

namespace lm {

using ProgressCallback = std::function<void(size_t epoch, float train_loss, float val_loss, float lr)>;

class LanguageModelTrainer {
public:
    // Constructor
    LanguageModelTrainer(const BPETokenizer& tokenizer,
                        size_t embedding_dim,
                        size_t hidden_dim,
                        size_t num_layers);

    // Training methods
    void train(const std::vector<std::string>& corpus,
              size_t epochs,
              size_t batch_size,
              size_t sequence_length,
              float validation_split = 0.1f,
              size_t validation_freq = 1,
              size_t early_stopping_patience = 0,
              const ProgressCallback& callback = nullptr);

    void train_on_conversations(
        const std::vector<Conversation>& conversations,
        size_t epochs,
        size_t batch_size,
        size_t sequence_length,
        size_t context_turns = 5,
        float validation_split = 0.1f,
        size_t validation_freq = 1,
        size_t early_stopping_patience = 0,
        const ProgressCallback& callback = nullptr);

    // Evaluation methods
    float evaluate(const std::vector<std::string>& corpus,
                  size_t batch_size,
                  size_t sequence_length);

    std::pair<float, float> test(const std::vector<std::string>& corpus,
                                size_t batch_size,
                                size_t sequence_length);

    // Generation methods
    std::string generate(const std::string& prompt,
                        size_t max_length,
                        Sampler& sampler,
                        size_t sequence_length);

    std::vector<std::string> generate_batch(const std::vector<std::string>& prompts,
                                           size_t max_length,
                                           Sampler& sampler,
                                           size_t sequence_length,
                                           size_t batch_size);

    std::string continue_conversation(
        const Conversation& conversation,
        Sampler& sampler,
        size_t max_length = 50,
        size_t sequence_length = 256,
        size_t context_turns = 5);

    // Utility methods
    std::vector<std::string> prepare_conversation_corpus(
        const std::vector<Conversation>& conversations,
        size_t context_turns,
        size_t max_sequence_length = 0);

    // Model persistence
    void save_model(const std::string& path);
    void load_model(const std::string& path);

    // Get training history
    const std::vector<float>& train_loss_history() const { return train_loss_history_; }
    const std::vector<float>& val_loss_history() const { return val_loss_history_; }

    // Public methods for model access
    void eval() {
        model_.eval();
    }

    void train_mode() {
        model_.train();
    }

    inline Tensor forward(const Tensor& input) {
        return model_.forward(input);
    }

    inline std::vector<Tensor> get_parameters() const {
        return model_.parameters();
    }

    inline Tensor prepare_input_batch(const std::vector<std::string>& texts, size_t sequence_length) {
        return prepare_batch(texts, sequence_length);
    }

    inline size_t get_parameter_count() const {
        return model_.parameters().size();
    }

    inline std::vector<int> generate_and_return_tokens(const std::string& prompt, 
                                          size_t max_length, 
                                          Sampler& sampler,
                                          size_t sequence_length) {
        return generate_tokens(prompt, max_length, sampler, sequence_length);
    }

private:
    // Private methods
    std::pair<std::vector<std::string>, std::vector<std::string>> split_data(
        const std::vector<std::string>& corpus, float validation_split);
    
    float run_validation(const std::vector<std::string>& validation_data,
                        size_t batch_size,
                        size_t sequence_length);
    
    Tensor prepare_batch(const std::vector<std::string>& texts,
                       size_t sequence_length);
    
    Tensor prepare_inference_batch(const std::vector<std::vector<int>>& token_sequences,
                                 size_t sequence_length);
    
    float compute_loss(const Tensor& logits, const Tensor& targets);
    
    std::vector<int> generate_tokens(const std::string& prompt,
                                   size_t max_length,
                                   Sampler& sampler,
                                   size_t sequence_length);

    // Member variables
    const BPETokenizer& tokenizer_;
    LanguageModel model_;
    AdamOptimizer optimizer_;
    std::vector<float> train_loss_history_;
    std::vector<float> val_loss_history_;
};

} // namespace lm
