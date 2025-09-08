// conversation_model.cpp
#include "conversation_model.hpp"
#include <algorithm>
#include <sstream>

namespace lm {

ConversationModel::ConversationModel(size_t vocab_size, size_t d_model, 
                                   size_t n_layers, size_t n_heads, 
                                   size_t d_ff, float dropout) {
    transformer_ = std::make_unique<TransformerModel>(vocab_size, d_model, n_layers, 
                                                     n_heads, d_ff, dropout);
}

void ConversationModel::train(const std::vector<std::string>& conversations) {
    for (const auto& conversation : conversations) {
        // Tokenize the conversation
        auto tokens = tokenizer_->encode(conversation);
        
        if (tokens.size() < 2) continue;
        
        // Create input and target sequences
        std::vector<TokenID> input_tokens(tokens.begin(), tokens.end() - 1);
        std::vector<TokenID> target_tokens(tokens.begin() + 1, tokens.end());
        
        // Training step
        transformer_->train_step(input_tokens, target_tokens);
    }
}

std::string ConversationModel::generate_response(const std::string& input, 
                                               const std::vector<std::string>& conversation_history) {
    // Format the input
    std::string formatted_input = format_input(input, conversation_history);
    
    // Tokenize
    auto tokens = tokenizer_->encode(formatted_input);
    
    // Generate continuation
    auto generated_tokens = transformer_->generate(tokens, 100, 0.8);
    
    // Decode
    std::string response = tokenizer_->decode(generated_tokens);
    
    // Extract only the new response (remove the input)
    if (response.find(formatted_input) == 0) {
        response = response.substr(formatted_input.length());
    }
    
    // Trim any special tokens
    size_t pos = response.find("<|endoftext|>");
    if (pos != std::string::npos) {
        response = response.substr(0, pos);
    }
    
    return response;
}

std::string ConversationModel::format_conversation(const std::vector<std::string>& turns) {
    std::stringstream ss;
    for (size_t i = 0; i < turns.size(); i++) {
        if (i % 2 == 0) {
            ss << "<|user|>" << turns[i] << "<|endoftext|>";
        } else {
            ss << "<|assistant|>" << turns[i] << "<|endoftext|>";
        }
    }
    return ss.str();
}

std::string ConversationModel::format_input(const std::string& input, 
                                          const std::vector<std::string>& conversation_history) {
    std::stringstream ss;
    
    // Add conversation history
    for (const auto& turn : conversation_history) {
        ss << turn;
    }
    
    // Add current input
    ss << "<|user|>" << input << "<|endoftext|><|assistant|>";
    
    return ss.str();
}

bool ConversationModel::save_model(const std::string& path) {
    return transformer_->save(path);
}

bool ConversationModel::load_model(const std::string& path) {
    return transformer_->load(path);
}

} // namespace lm
