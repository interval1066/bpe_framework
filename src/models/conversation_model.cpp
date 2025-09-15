// Enhanced conversation_model.cpp
#include "lm/models/conversation_model.hpp"
#include <algorithm>
#include <sstream>
#include <iostream>
#include <stdexcept>

namespace lm {

ConversationModel::ConversationModel(size_t vocab_size, size_t d_model, 
                                   size_t n_layers, size_t n_heads, 
                                   size_t d_ff, float dropout) {
    transformer_ = std::make_unique<TransformerModel>(vocab_size, d_model, n_layers, 
                                                     n_heads, d_ff, dropout);
}

void ConversationModel::train(const std::vector<std::string>& conversations) {
    // Validate tokenizer is set
    if (!tokenizer_) {
        throw std::runtime_error("Tokenizer not set before training");
    }
    
    size_t max_seq_length = 0;
    std::vector<std::vector<TokenID>> all_tokens;
    
    // First pass: tokenize all conversations and find max length
    for (const auto& conversation : conversations) {
        auto tokens = tokenizer_->encode(conversation);
        if (tokens.size() < 2) continue;
        
        max_seq_length = std::max(max_seq_length, tokens.size());
        all_tokens.push_back(tokens);
    }
    
    // Second pass: pad sequences and train
    for (const auto& tokens : all_tokens) {
        if (tokens.size() < 2) continue;
        
        // Create input and target sequences
        // Input: all tokens except the last one
        std::vector<TokenID> input_tokens(tokens.begin(), tokens.end() - 1);
        // Target: all tokens except the first one (shifted by one)
        std::vector<TokenID> target_tokens(tokens.begin() + 1, tokens.end());
        
        // Pad both sequences to the same length (max_seq_length - 1)
        if (input_tokens.size() < max_seq_length - 1) {
            input_tokens.resize(max_seq_length - 1, pad_token_id_);
        }
        
        if (target_tokens.size() < max_seq_length - 1) {
            target_tokens.resize(max_seq_length - 1, pad_token_id_);
        }
        
        // Training step
        try {
            transformer_->train_step(input_tokens, target_tokens);
        } catch (const std::exception& e) {
            std::cerr << "Error during training step: " << e.what() << std::endl;
            std::cerr << "Input size: " << input_tokens.size() 
                      << ", Target size: " << target_tokens.size() << std::endl;
            throw;
        }
    }
}

std::string ConversationModel::generate_response(const std::string& user_input) {
    // Validate tokenizer is set
    if (!tokenizer_) {
        throw std::runtime_error("Tokenizer not set before generation");
    }
    
    // Add user message to context
    context_manager_->add_user_message(user_input);
    
    // Get the full context
    std::string context = context_manager_->get_context();
    
    // Add assistant role tag to prompt the model
    context += "<|assistant|>";
    
    // Tokenize context
    auto tokens = tokenizer_->encode(context);
    
    // Ensure we have tokens to generate from
    if (tokens.empty()) {
        return "I'm not sure how to respond to that.";
    }
    
    // Generate continuation
    auto generated_tokens = transformer_->generate(tokens, 100, 0.8);
    
    // Decode
    std::string response = tokenizer_->decode(generated_tokens);
    
    // Remove the context part to get just the new response
    if (response.find(context) == 0) {
        response = response.substr(context.length());
    }
    
    // Remove any trailing endoftext tokens
    size_t end_pos = response.find("<|endoftext|>");
    if (end_pos != std::string::npos) {
        response = response.substr(0, end_pos);
    }
    
    // Add assistant response to context
    context_manager_->add_assistant_message(response);
    
    return response;
}

void ConversationModel::clear_context() {
    context_manager_->clear();
    if (!system_prompt_.empty()) {
        context_manager_->add_system_message(system_prompt_);
    }
}

void ConversationModel::set_system_prompt(const std::string& prompt) {
    system_prompt_ = prompt;
    clear_context(); // Reset context with new system prompt
}

size_t ConversationModel::get_context_token_count() const {
    return context_manager_->get_token_count();
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

bool ConversationModel::save_model(const std::string& path) {
    return transformer_->save(path);
}

bool ConversationModel::load_model(const std::string& path) {
    return transformer_->load(path);
}

} // namespace lm

