// context_manager.cpp
#include "context_manager.hpp"
#include "lm/tokenizer/bpe_tokenizer.hpp"
#include <algorithm>

namespace lm {

ContextManager::ContextManager(size_t max_context_tokens, size_t max_turns)
    : max_context_tokens(max_context_tokens), max_turns(max_turns), current_token_count(0) {}

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
    // Tokenize to count tokens (in a real implementation, you'd use your tokenizer)
    // For now, we'll use a simple approximation
    size_t token_count = content.size() / 4; // Rough approximation
    
    conversation_turns.push_back({role, content, token_count});
    current_token_count += token_count;
    
    // Add role tokens
    current_token_count += 5; // Approximate token count for role tags
    
    prune_old_messages();
}

void ContextManager::prune_old_messages() {
    while (current_token_count > max_context_tokens && conversation_turns.size() > 1) {
        // Remove the oldest turn
        const auto& oldest_turn = conversation_turns.front();
        current_token_count -= oldest_turn.token_count;
        current_token_count -= 5; // Role tags
        
        conversation_turns.pop_front();
    }
    
    // Also respect max turns limit
    while (conversation_turns.size() > max_turns) {
        const auto& oldest_turn = conversation_turns.front();
        current_token_count -= oldest_turn.token_count;
        current_token_count -= 5; // Role tags
        
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
    // In a real implementation, you'd tokenize the context
    // For now, return empty vector
    return {};
}

void ContextManager::clear() {
    conversation_turns.clear();
    current_token_count = 0;
}

} // namespace lm
