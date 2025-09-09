// Enhanced conversation_model.cpp
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

std::string ConversationModel::generate_response(const std::string& user_input) {
    // Add user message to context
    context_manager_->add_user_message(user_input);
    
    // Get the full context
    std::string context = context_manager_->get_context();
    
    // Add assistant role tag to prompt the model
    context += "<|assistant|>";
    
    // Tokenize context
    auto tokens = tokenizer_->encode(context);
    
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

