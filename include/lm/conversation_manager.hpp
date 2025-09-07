// include/lm/conversation_manager.hpp
#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <mutex>
#include "conversation.hpp"

namespace lm {

class ConversationManager {
public:
    ConversationManager();
    ~ConversationManager();
    
    // Create a new conversation
    std::string create_conversation(const std::string& title = "");
    
    // Get a conversation by ID
    std::shared_ptr<Conversation> get_conversation(const std::string& id);
    
    // Get all conversation IDs
    std::vector<std::string> list_conversations() const;
    
    // Add a message to a conversation
    void add_message(const std::string& conversation_id, 
                     const std::string& role, 
                     const std::string& content);
    
    // Get conversation history
    std::vector<ConversationTurn> get_history(const std::string& conversation_id) const;
    
    // Save conversations to disk
    bool save_conversations(const std::string& path) const;
    
    // Load conversations from disk
    bool load_conversations(const std::string& path);
    
    // Delete a conversation
    bool delete_conversation(const std::string& id);
    
    // Set conversation title
    void set_title(const std::string& conversation_id, const std::string& title);
    
    // Get conversation title
    std::string get_title(const std::string& conversation_id) const;
    
    // Get conversation metadata
    std::map<std::string, std::string> get_metadata(const std::string& conversation_id) const;
    
    // Update conversation metadata
    void update_metadata(const std::string& conversation_id, 
                         const std::map<std::string, std::string>& metadata);
    
    // Clear all conversations
    void clear();
    
    // Get number of conversations
    size_t count() const;

private:
    std::unordered_map<std::string, std::shared_ptr<Conversation>> conversations_;
    mutable std::mutex mutex_;
    
    // Generate a unique ID for conversations
    std::string generate_id() const;
};

} // namespace lm

