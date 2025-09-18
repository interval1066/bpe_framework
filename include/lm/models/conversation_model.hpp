// conversation_model.hpp
#pragma once

#include "lm/models/transformer_model.hpp"
#include "lm/tokenizer/bpe_tokenizer.hpp"
#include "lm/data/training_data.hpp"
#include "lm/context_manager.hpp"
#include <string>
#include <vector>
#include <memory>
#include <limits>

namespace lm {

class ConversationModel {
public:
    ConversationModel(size_t vocab_size, 
                     size_t d_model = 512, 
                     size_t n_layers = 6, 
                     size_t n_heads = 8,
                     size_t d_ff = 2048,
                     float dropout = 0.1);
    
    // Train the model with a dataset
    void train(const TrainingDataset& dataset, size_t num_epochs = 50, float learning_rate = 0.01f);
    
    // Generate a response
    std::string generate_response(const std::string& user_input);
    
    // Context management
    void clear_context();
    void set_system_prompt(const std::string& prompt);
    size_t get_context_token_count() const;
    
    // Save and load
    bool save_checkpoint(const std::string& filename);
    bool load_checkpoint(const std::string& filename);
    
    // Set tokenizer
    void set_tokenizer(std::shared_ptr<BPETokenizer> tokenizer) { 
        tokenizer_ = tokenizer; 
        context_manager_ = std::make_unique<ContextManager>(2048, 20);
    }

    inline size_t vocab_size() const {
        return transformer_->vocab_size();
    }

    void set_verbose(bool verbose) { verbose_ = verbose; }
    void set_max_response_length(size_t length) { max_response_length_ = length; }

private:
    std::shared_ptr<BPETokenizer> tokenizer_;
    std::unique_ptr<TransformerModel> transformer_;
    std::unique_ptr<ContextManager> context_manager_;

    std::string system_prompt_;
    TokenID pad_token_id_;
    bool verbose_ = false;
    size_t max_response_length_ = 20;
    
    // Training metadata
    size_t training_epochs_ = 0;
    float best_loss_ = std::numeric_limits<float>::max();
    std::string training_timestamp_;
    
    // Helper method for timestamps
    std::string get_current_timestamp() const;
};

} // namespace lm

