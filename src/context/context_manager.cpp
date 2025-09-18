// src/context/context_manager.cpp
#include "lm/context_manager.hpp"
#include "lm/tokenizer/bpe_tokenizer.hpp"
#include <algorithm>
#include <memory>
#include <stdexcept>

namespace lm {

ContextManager::ContextManager(size_t max_context_tokens, size_t max_turns)
    : max_context_tokens(max_context_tokens), max_turns(max_turns), current_token_count(0) {
    // Initialize with a default tokenizer (will be replaced when ConversationModel sets it)
    tokenizer_ = std::make_shared<BPETokenizer>();
}

void ContextManager::set_tokenizer(std::shared_ptr<BPETokenizer> tokenizer) {
    tokenizer_ = tokenizer;
}

void ContextManager::add_user_message(const std::string& message) {
    add_message("user", message);
}

void ContextManager::add_assistant_message(const std::string& message) {
    add_message("assistant", message);
}

void ContextManager::add_system_message(const std::string& message) {
    add_message("system", message);
}

void ContextManager::add_message(const std::string& role, const std::string& content) {
    if (!tokenizer_) {
        throw std::runtime_error("Tokenizer not set in ContextManager");
    }
    
    // Tokenize the content to get accurate token count
    auto tokens = tokenizer_->encode(content);
    size_t token_count = tokens.size();
    
    // Add role tokens (e.g., "<|user|>")
    std::string role_tag = "<|" + role + "|>";
    auto role_tokens = tokenizer_->encode(role_tag);
    token_count += role_tokens.size();
    
    // Add endoftext tokens
    auto end_tokens = tokenizer_->encode("<|endoftext|>");
    token_count += end_tokens.size();
    
    conversation_turns.push_back({role, content, token_count});
    current_token_count += token_count;
    
    prune_old_messages();
}

void ContextManager::prune_old_messages() {
    while (current_token_count > max_context_tokens && conversation_turns.size() > 1) {
        // Remove the oldest turn
        const auto& oldest_turn = conversation_turns.front();
        current_token_count -= oldest_turn.token_count;
        conversation_turns.pop_front();
    }
    
    // Also respect max turns limit
    while (conversation_turns.size() > max_turns) {
        const auto& oldest_turn = conversation_turns.front();
        current_token_count -= oldest_turn.token_count;
        conversation_turns.pop_front();
    }
}

std::string ContextManager::get_context() const {
    std::string context;
    
    for (const auto& turn : conversation_turns) {
        context += "<|" + turn.role + "|>" + turn.content + "<|endoftext|>";
    }
    
    return context;
}

std::vector<TokenID> ContextManager::get_context_tokens() const {
    if (!tokenizer_) {
        throw std::runtime_error("Tokenizer not set in ContextManager");
    }
    
    std::string context = get_context();
    return tokenizer_->encode(context);
}

void ContextManager::clear() {
    conversation_turns.clear();
    current_token_count = 0;
}

} // namespace lm

