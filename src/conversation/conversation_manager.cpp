// src/conversation/conversation_manager.cpp
#include "lm/conversation/conversation_manager.hpp"
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <fstream>

namespace lm {

ConversationManager::ConversationManager(std::shared_ptr<TransformerModel> model,
                                       std::shared_ptr<BPETokenizer> tokenizer,
                                       std::unique_ptr<Sampler> sampler)
    : model_(model), tokenizer_(tokenizer), sampler_(std::move(sampler)),
      max_history_length_(10) {}

std::string ConversationManager::generate_response(const std::string& input, 
                                                 size_t max_length,
                                                 float temperature) {
    // Add user input to history
    add_to_history(input, true);
    
    // Format the conversation for the model
    auto input_tokens = format_conversation();
    
    // Generate response tokens
    std::vector<unsigned int> response_tokens;
    
    // Convert to the format expected by the model (vector<unsigned int>)
    std::vector<unsigned int> model_input(input_tokens.begin(), input_tokens.end());
    auto current_tokens = model_input;
    
    for (size_t i = 0; i < max_length; i++) {
        // Get model predictions
        auto logits_vector = model_->forward(current_tokens);
        
        // Convert logits vector to Tensor
        // Assuming logits are shaped as [seq_len, vocab_size]
        size_t seq_len = current_tokens.size();
        size_t vocab_size = logits_vector.size() / seq_len;
        
        Tensor logits({seq_len, vocab_size}, false);
        for (size_t s = 0; s < seq_len; s++) {
            for (size_t v = 0; v < vocab_size; v++) {
                logits(s, v) = logits_vector[s * vocab_size + v];
            }
        }
        
        // Get the last token logits
        auto last_token_logits = logits.slice(seq_len - 1, 1, 0);
        
        // Sample next token
        int next_token = sampler_->sample(last_token_logits);
        
        // Check for end of sequence token (assuming 2 is EOS)
        if (next_token == 2) {
            break;
        }
        
        response_tokens.push_back(static_cast<unsigned int>(next_token));
        current_tokens.push_back(static_cast<unsigned int>(next_token));
    }
    
    // Decode the response
    std::string response = tokenizer_->decode(response_tokens);
    
    // Add AI response to history
    add_to_history(response, false);
    
    return response;
}

void ConversationManager::add_to_history(const std::string& utterance, bool is_user) {
    std::string prefix = is_user ? "User: " : "AI: ";
    conversation_history_.push_back(prefix + utterance);
    
    // Limit history length
    if (conversation_history_.size() > max_history_length_ * 2) {
        // Keep the most recent exchanges
        conversation_history_.erase(conversation_history_.begin(), 
                                  conversation_history_.begin() + 2);
    }
}

void ConversationManager::clear_history() {
    conversation_history_.clear();
}

std::vector<std::string> ConversationManager::get_history() const {
    return conversation_history_;
}

void ConversationManager::set_sampler(std::unique_ptr<Sampler> sampler) {
    sampler_ = std::move(sampler);
}

std::vector<int> ConversationManager::format_conversation() const {
    std::vector<int> tokens;
    
    // Add special start token (assuming 1 is BOS)
    tokens.push_back(1);
    
    // Add conversation history
    for (const auto& utterance : conversation_history_) {
        auto utterance_tokens = tokenizer_->encode(utterance);
        // Convert unsigned int tokens to int tokens
        std::vector<int> int_tokens(utterance_tokens.begin(), utterance_tokens.end());
        tokens.insert(tokens.end(), int_tokens.begin(), int_tokens.end());
        
        // Add separator token (assuming 3 is SEP)
        tokens.push_back(3);
    }
    
    return tokens;
}

void ConversationManager::save_conversation(const std::string& path) const {
    std::ofstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for saving conversation: " + path);
    }
    
    for (const auto& message : conversation_history_) {
        file << message << "\n";
    }
}

void ConversationManager::load_conversation(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open conversation file: " + path);
    }
    
    clear_history();
    
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            // Determine if it's a user or AI message
            bool is_user = line.find("User: ") == 0;
            std::string content = is_user ? line.substr(6) : line.substr(4);
            
            add_to_history(content, is_user);
        }
    }
}

} // namespace lm

