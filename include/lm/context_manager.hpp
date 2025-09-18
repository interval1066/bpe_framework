// include/lm/context_manager.hpp
#pragma once

#include <vector>
#include <string>
#include <deque>
#include <memory>
#include "lm/tokenizer/bpe_tokenizer.hpp"

namespace lm {

class ContextManager {
public:
    ContextManager(size_t max_context_tokens = 2048, 
                  size_t max_turns = 20);
    
    void set_tokenizer(std::shared_ptr<BPETokenizer> tokenizer);
    void add_user_message(const std::string& message);
    void add_assistant_message(const std::string& message);
    void add_system_message(const std::string& message);
    
    std::string get_context() const;
    std::vector<TokenID> get_context_tokens() const;
    
    void clear();
    void prune_old_messages();
    
    size_t get_token_count() const { return current_token_count; }
    size_t get_turn_count() const { return conversation_turns.size(); }

private:
    struct ConversationTurn {
        std::string role;  // "user", "assistant", or "system"
        std::string content;
        size_t token_count;
    };
    
    std::deque<ConversationTurn> conversation_turns;
    size_t max_context_tokens;
    size_t max_turns;
    size_t current_token_count;
    std::shared_ptr<BPETokenizer> tokenizer_;
    
    void add_message(const std::string& role, const std::string& content);
};

} // namespace lm
