// conversation_model.hpp
#pragma once

#include "transformer_model.hpp"
#include "bpe_tokenizer.hpp"
#include <string>
#include <vector>

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
    
    // Generate a response
    std::string generate_response(const std::string& input, 
                                 const std::vector<std::string>& conversation_history = {});
    
    // Save and load
    bool save_model(const std::string& path);
    bool load_model(const std::string& path);
    
    // Set tokenizer
    void set_tokenizer(std::shared_ptr<BPETokenizer> tokenizer) { tokenizer_ = tokenizer; }

private:
    std::shared_ptr<BPETokenizer> tokenizer_;
    std::unique_ptr<TransformerModel> transformer_;
    
    // Format conversation for training
    std::string format_conversation(const std::vector<std::string>& turns);
    
    // Format input for inference
    std::string format_input(const std::string& input, 
                            const std::vector<std::string>& conversation_history);
};

} // namespace lm

