// Enhanced conversation_model.hpp
#pragma once

#include "lm/models/transformer_model.hpp"
#include "bpe_tokenizer.hpp"
#include "context_manager.hpp"
#include <string>
#include <vector>
#include <memory>

namespace lm {

class ConversationModel {
public:
    ConversationModel(size_t vocab_size, 
                     size_t d_model = 512, 
                     size_t n_layers = 6, 
                     size_t n_heads = 8,
                     size_t d_ff = 2048,
                     float dropout = 0.1);
    
    // Train the model
    void train(const std::vector<std::string>& conversations);
    
    // Generate a response with context management
    std::string generate_response(const std::string& user_input);
    
    // Context management
    void clear_context();
    void set_system_prompt(const std::string& prompt);
    size_t get_context_token_count() const;
    
    // Save and load
    bool save_model(const std::string& path);
    bool load_model(const std::string& path);
    
    // Set tokenizer
    void set_tokenizer(std::shared_ptr<BPETokenizer> tokenizer) { 
        tokenizer_ = tokenizer; 
        context_manager_ = std::make_unique<ContextManager>(2048, 20);
    }

    inline size_t vocab_size() const {
        return transformer_->vocab_size();
    }

private:
    std::shared_ptr<BPETokenizer> tokenizer_;
    std::unique_ptr<TransformerModel> transformer_;
    std::unique_ptr<ContextManager> context_manager_;
    std::string system_prompt_;
    TokenID pad_token_id_;

    // Format conversation for training
    std::string format_conversation(const std::vector<std::string>& turns);
};

} // namespace lm

