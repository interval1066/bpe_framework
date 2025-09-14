// src/test_conversation.cpp
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include "lm/conversation/conversation_manager.hpp"
#include "lm/models/transformer_model.hpp"
#include "lm/tokenizer/bpe_tokenizer.hpp"
#include "lm/generation/greedy_sampler.hpp"

void test_conversation_manager() {
    std::cout << "=== Testing Conversation Manager ===" << std::endl;
    
    // Create necessary components for ConversationManager
    auto dummy_model = std::make_shared<lm::TransformerModel>(1000, 512, 6, 8, 2048, 0.1f);
    auto tokenizer = std::make_shared<lm::BPETokenizer>();
    auto sampler = std::make_unique<lm::GreedySampler>();
    
    lm::ConversationManager manager(dummy_model, tokenizer, std::move(sampler));
    
    // Add messages to conversation
    manager.add_to_history("What's the weather like today?", true);
    manager.add_to_history("It's sunny and 75 degrees.", false);
    manager.add_to_history("Should I bring an umbrella?", true);
    
    // Test getting history
    auto history = manager.get_history();
    std::cout << "Conversation history:" << std::endl;
    for (size_t i = 0; i < history.size(); ++i) {
        std::cout << "  " << i << ": " << history[i] << std::endl;
    }
    
    // Test clearing history
    std::cout << "Clearing conversation history..." << std::endl;
    manager.clear_history();
    std::cout << "History size after clearing: " << manager.get_history().size() << std::endl;
    
    // Add new messages
    manager.add_to_history("Hello, how are you?", true);
    manager.add_to_history("I'm doing well, thank you!", false);
    
    // Test saving and loading conversation
    std::cout << "Saving conversation..." << std::endl;
    manager.save_conversation("test_conversation.txt");
    
    // Create a new manager and load the conversation
    auto dummy_model2 = std::make_shared<lm::TransformerModel>(1000, 512, 6, 8, 2048, 0.1f);
    auto tokenizer2 = std::make_shared<lm::BPETokenizer>();
    auto sampler2 = std::make_unique<lm::GreedySampler>();
    
    lm::ConversationManager loaded_manager(dummy_model2, tokenizer2, std::move(sampler2));
    std::cout << "Loading conversation..." << std::endl;
    loaded_manager.load_conversation("test_conversation.txt");
    
    // Verify the loaded conversation
    auto loaded_history = loaded_manager.get_history();
    std::cout << "Loaded conversation history:" << std::endl;
    for (size_t i = 0; i < loaded_history.size(); ++i) {
        std::cout << "  " << i << ": " << loaded_history[i] << std::endl;
    }
    
    std::cout << "=== Conversation Manager Test Complete ===\n" << std::endl;
}

void test_response_generation() {
    std::cout << "=== Testing Response Generation ===" << std::endl;
    
    // Create necessary components for ConversationManager
    auto dummy_model = std::make_shared<lm::TransformerModel>(1000, 512, 6, 8, 2048, 0.1f);
    auto tokenizer = std::make_shared<lm::BPETokenizer>();
    auto sampler = std::make_unique<lm::GreedySampler>();
    
    lm::ConversationManager manager(dummy_model, tokenizer, std::move(sampler));
    
    // Add some context
    manager.add_to_history("Hello, how are you?", true);
    manager.add_to_history("I'm doing well, thank you!", false);
    
    try {
        // Try to generate a response (this will fail with the dummy model, but test the flow)
        std::cout << "Generating response..." << std::endl;
        std::string response = manager.generate_response("What's the weather like today?");
        std::cout << "Generated response: " << response << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Response generation failed (expected with dummy model): " << e.what() << std::endl;
    }
    
    std::cout << "=== Response Generation Test Complete ===\n" << std::endl;
}

int main() {
    std::cout << "Starting Conversation Manager Tests\n" << std::endl;
    
    try {
        test_conversation_manager();
        test_response_generation();
        
        std::cout << "All tests completed successfully!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

