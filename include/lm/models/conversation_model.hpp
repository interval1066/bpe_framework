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
    void train(const TrainingDataset& dataset, 
               size_t num_epochs = 50, 
               float learning_rate = 0.001f, // Reduced default
               const std::string& resume_checkpoint = "");

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
    void set_tokenizer(std::shared_ptr<BPETokenizer> tokenizer);
    
    // Utility functions
    bool checkpoint_exists(const std::string& filename) const;
    std::string find_latest_checkpoint() const;
    void set_checkpoint_interval(size_t interval) { checkpoint_interval_ = interval; }

    // Training configuration
    void set_gradient_clipping(float max_norm) { max_grad_norm_ = max_norm; }
    void set_learning_rate_decay(float decay) { lr_decay_ = decay; }
    void set_weight_decay(float decay) { weight_decay_ = decay; }

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
    size_t checkpoint_interval_ = 10;
    
    // Training configuration
    float max_grad_norm_ = 1.0f;      // Gradient clipping
    float lr_decay_ = 0.95f;          // Learning rate decay
    float weight_decay_ = 0.01f;      // Weight decay for regularization
    
    // Training metadata
    size_t training_epochs_ = 0;
    float best_loss_ = std::numeric_limits<float>::max();
    std::string training_timestamp_;
    
    // Helper method for timestamps
    std::string get_current_timestamp() const;
    
    // Improved training helpers
    float calculate_learning_rate(size_t epoch, size_t total_epochs, float base_lr) const;
    bool validate_training_example(const std::vector<TokenID>& input, 
                                  const std::vector<TokenID>& target) const;
    void apply_gradient_clipping(); // Assuming transformer has access to gradients
};

} // namespace lm
