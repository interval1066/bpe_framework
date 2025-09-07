// src/test_conversation.cpp
#include <iostream>
#include <string>
#include <vector>
#include "lm/conversation_manager.hpp"
#include "lm/conversation.hpp"

void print_conversation(const lm::Conversation& conv, const std::string& id) {
    std::cout << "=== Conversation " << id << " ===" << std::endl;
    std::cout << "Domain: " << conv.domain << std::endl;
    std::cout << "Language: " << conv.language << std::endl;
    std::cout << "Turns: " << conv.turns.size() << std::endl;
    std::cout << "Duration: " << conv.duration() << " seconds" << std::endl;
    
    for (size_t i = 0; i < conv.turns.size(); ++i) {
        const auto& turn = conv.turns[i];
        auto time = std::chrono::system_clock::to_time_t(turn.timestamp);
        std::cout << "[" << i << "] " << std::ctime(&time) 
                  << lm::speaker_type_to_string(turn.speaker) 
                  << ": " << turn.text << std::endl;
    }
    std::cout << std::endl;
}

void test_conversation_basic() {
    std::cout << "=== Testing Basic Conversation Functionality ===" << std::endl;
    
    // Create a conversation
    lm::Conversation conv("general_chat", "en");
    conv.add_turn(lm::SpeakerType::USER, "Hello, how are you?");
    conv.add_turn(lm::SpeakerType::ASSISTANT, "I'm doing well, thank you!");
    conv.add_turn(lm::SpeakerType::USER, "What's the weather like today?");
    
    // Test basic properties
    std::cout << "Conversation has " << conv.size() << " turns" << std::endl;
    std::cout << "Duration: " << conv.duration() << " seconds" << std::endl;
    std::cout << "Domain: " << conv.domain << std::endl;
    
    // Test last turn access
    try {
        auto& last_turn = conv.last_turn();
        std::cout << "Last turn: " << last_turn.text << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Error accessing last turn: " << e.what() << std::endl;
    }
    
    // Test clearing
    std::cout << "Clearing conversation..." << std::endl;
    conv.clear();
    std::cout << "After clearing: " << conv.size() << " turns" << std::endl;
    
    std::cout << "=== Basic Conversation Test Complete ===\n" << std::endl;
}

void test_conversation_manager() {
    std::cout << "=== Testing Conversation Manager ===" << std::endl;
    
    lm::ConversationManager manager;
    
    // Create conversations
    std::string conv1 = manager.create_conversation("Weather Discussion");
    std::string conv2 = manager.create_conversation("Technical Support");
    
    std::cout << "Created conversations: " << conv1 << " and " << conv2 << std::endl;
    
    // Add messages to first conversation
    manager.add_message(conv1, "user", "What's the weather like today?");
    manager.add_message(conv1, "assistant", "It's sunny and 75 degrees.");
    manager.add_message(conv1, "user", "Should I bring an umbrella?");
    
    // Add messages to second conversation
    manager.add_message(conv2, "user", "My computer won't turn on.");
    manager.add_message(conv2, "assistant", "Have you tried checking the power cable?");
    
    // List all conversations
    auto conversations = manager.list_conversations();
    std::cout << "Total conversations: " << conversations.size() << std::endl;
    
    for (const auto& id : conversations) {
        std::cout << "Conversation ID: " << id 
                  << ", Title: " << manager.get_title(id) << std::endl;
        
        auto conv_ptr = manager.get_conversation(id);
        if (conv_ptr) {
            std::cout << "  Turns: " << conv_ptr->size() << std::endl;
        }
    }
    
    // Test getting history
    try {
        auto history = manager.get_history(conv1);
        std::cout << "\nHistory for conversation " << conv1 << ":" << std::endl;
        for (size_t i = 0; i < history.size(); ++i) {
            std::cout << "  " << i << ": " 
                      << lm::speaker_type_to_string(history[i].speaker) 
                      << ": " << history[i].text << std::endl;
        }
    } catch (const std::exception& e) {
        std::cout << "Error getting history: " << e.what() << std::endl;
    }
    
    // Test metadata operations
    manager.set_title(conv1, "Updated Weather Chat");
    std::cout << "Updated title: " << manager.get_title(conv1) << std::endl;
    
    std::map<std::string, std::string> metadata = {
        {"priority", "high"},
        {"category", "weather"}
    };
    manager.update_metadata(conv1, metadata);
    
    auto retrieved_metadata = manager.get_metadata(conv1);
    std::cout << "Metadata: " << std::endl;
    for (const auto& pair : retrieved_metadata) {
        std::cout << "  " << pair.first << ": " << pair.second << std::endl;
    }
    
    // Test deletion
    std::cout << "Deleting conversation " << conv2 << std::endl;
    bool deleted = manager.delete_conversation(conv2);
    std::cout << "Deletion " << (deleted ? "successful" : "failed") << std::endl;
    std::cout << "Remaining conversations: " << manager.count() << std::endl;
    
    std::cout << "=== Conversation Manager Test Complete ===\n" << std::endl;
}

void test_serialization() {
    std::cout << "=== Testing Serialization ===" << std::endl;
    
    lm::ConversationManager manager;
    
    // Create a conversation with some messages
    std::string conv_id = manager.create_conversation("Serialization Test");
    manager.add_message(conv_id, "user", "This is a test message.");
    manager.add_message(conv_id, "assistant", "This is a test response.");
    manager.add_message(conv_id, "user", "Will this be saved correctly?");
    
    // Save to file
    std::string filename = "test_conversations.bin";
    bool saved = manager.save_conversations(filename);
    std::cout << "Save " << (saved ? "successful" : "failed") << std::endl;
    
    // Create a new manager and load from file
    lm::ConversationManager loaded_manager;
    bool loaded = loaded_manager.load_conversations(filename);
    std::cout << "Load " << (loaded ? "successful" : "failed") << std::endl;
    
    if (loaded) {
        auto conversations = loaded_manager.list_conversations();
        std::cout << "Loaded conversations: " << conversations.size() << std::endl;
        
        for (const auto& id : conversations) {
            std::cout << "Conversation ID: " << id 
                      << ", Title: " << loaded_manager.get_title(id) << std::endl;
            
            auto history = loaded_manager.get_history(id);
            std::cout << "  Messages: " << history.size() << std::endl;
            
            for (const auto& turn : history) {
                std::cout << "    " << lm::speaker_type_to_string(turn.speaker) 
                          << ": " << turn.text << std::endl;
            }
        }
    }
    
    std::cout << "=== Serialization Test Complete ===\n" << std::endl;
}

void test_conversation_utils() {
    std::cout << "=== Testing Conversation Utilities ===" << std::endl;
    
    lm::Conversation conv("test", "en");
    conv.add_turn(lm::SpeakerType::USER, "Hello");
    conv.add_turn(lm::SpeakerType::ASSISTANT, "Hi there!");
    conv.add_turn(lm::SpeakerType::USER, "How are you?");
    conv.add_turn(lm::SpeakerType::ASSISTANT, "I'm fine, thanks!");
    conv.add_turn(lm::SpeakerType::USER, "What's new?");
    
    // Test text extraction
    std::string extracted = lm::conversation_utils::extract_text(conv.turns, 1, 4);
    std::cout << "Extracted text:\n" << extracted << std::endl;
    
    // Test training pair creation
    auto training_pair = lm::conversation_utils::create_training_pair(conv.turns, 2);
    std::cout << "Training context:\n" << training_pair.first << std::endl;
    std::cout << "Training target: " << training_pair.second << std::endl;
    
    // Test context window
    auto context_window = lm::conversation_utils::get_context_window(conv.turns, 3);
    std::cout << "Context window (last 3 turns):" << std::endl;
    for (const auto& turn : context_window) {
        std::cout << "  " << lm::speaker_type_to_string(turn.speaker) 
                  << ": " << turn.text << std::endl;
    }
    
    std::cout << "=== Conversation Utilities Test Complete ===\n" << std::endl;
}

int main() {
    std::cout << "Starting Conversation Manager Tests\n" << std::endl;
    
    try {
        test_conversation_basic();
        test_conversation_manager();
        test_serialization();
        test_conversation_utils();
        
        std::cout << "All tests completed successfully!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
