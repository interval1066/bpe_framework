// main.cpp
#include "lm/models/conversation_model.hpp"
#include "lm/tokenizer/bpe_tokenizer.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>
#include <memory>

// Helper function to get current timestamp
std::string get_current_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %X");
    return ss.str();
}

int main() {
    std::cout << "[" << get_current_timestamp() << "] Starting conversation model initialization..." << std::endl;
    
    try {
        // Initialize tokenizer
        std::cout << "[" << get_current_timestamp() << "] Creating BPE tokenizer..." << std::endl;
        auto tokenizer = std::make_shared<lm::BPETokenizer>();
        
        // Train or load tokenizer
        std::cout << "[" << get_current_timestamp() << "] Preparing training data for tokenizer..." << std::endl;
        std::vector<std::string> training_data = {
            "Hello, how are you?",
            "I'm doing well, thank you!",
            "What can I help you with today?",
            "The weather is nice today.",
            "I enjoy programming in C++.",
            "Machine learning is fascinating.",
            "Natural language processing enables computers to understand human language.",
            "This is a test of the tokenizer system.",
            "Reinforcement learning uses rewards to train agents.",
            "Deep learning models have many layers."
        };
        
        std::cout << "[" << get_current_timestamp() << "] Training tokenizer with " << training_data.size() << " examples..." << std::endl;
        tokenizer->train(training_data, 1000);
        std::cout << "[" << get_current_timestamp() << "] Tokenizer training completed. Vocabulary size: " << tokenizer->vocab_size() << std::endl;
        
        // Initialize conversation model with explicit parameters
        std::cout << "[" << get_current_timestamp() << "] Initializing conversation model..." << std::endl;
        lm::ConversationModel model(
            tokenizer->vocab_size(), // vocab_size
            512,                     // d_model
            6,                       // n_layers
            8,                       // n_heads
            2048,                    // d_ff
            0.1f                     // dropout
        );
        model.set_tokenizer(tokenizer);
        
        // Test the tokenizer with a simple input
        std::cout << "[" << get_current_timestamp() << "] Testing tokenizer..." << std::endl;
        std::string test_text = "Hello";
        std::vector<lm::TokenID> test_tokens = tokenizer->encode(test_text);
        std::string decoded_text = tokenizer->decode(test_tokens);
        
        std::cout << "[" << get_current_timestamp() << "] Tokenizer test:" << std::endl;
        std::cout << "  Input: '" << test_text << "'" << std::endl;
        std::cout << "  Tokens: ";
        for (lm::TokenID id : test_tokens) {
            std::cout << id << " ";
        }
        std::cout << std::endl;
        std::cout << "  Decoded: '" << decoded_text << "'" << std::endl;
        
        // Check if the token IDs are within the vocabulary size
        for (lm::TokenID id : test_tokens) {
            if (id >= tokenizer->vocab_size()) {
                std::cerr << "[" << get_current_timestamp() << "] ERROR: Token ID " << id 
                          << " is out of range (vocab size: " << tokenizer->vocab_size() << ")!" << std::endl;
                return 1;
            }
        }
        
        if (test_text != decoded_text) {
            std::cerr << "[" << get_current_timestamp() << "] ERROR: Tokenizer round-trip failed!" << std::endl;
            return 1;
        }
        
        // Train the model with simplified data
        std::cout << "[" << get_current_timestamp() << "] Preparing conversation training data..." << std::endl;
        
        // Use a single, very simple example to isolate the issue
        std::vector<std::string> simple_example = {
            "H"  // Single character
        };
        
        std::cout << "[" << get_current_timestamp() << "] Training conversation model with " << simple_example.size() << " examples..." << std::endl;
        
        try {
            model.train(simple_example);
            std::cout << "[" << get_current_timestamp() << "] Model training completed." << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[" << get_current_timestamp() << "] Error during training: " << e.what() << std::endl;
            
            // Try to get more information about the error
            std::cerr << "[" << get_current_timestamp() << "] Tokenizer vocab size: " << tokenizer->vocab_size() << std::endl;
            std::cerr << "[" << get_current_timestamp() << "] Model vocab size: " << model.vocab_size() << std::endl;
            
            return 1;
        }
        
        // Test with a simple input
        std::cout << "[" << get_current_timestamp() << "] Testing model with sample input..." << std::endl;
        std::string test_input = "Hello";
        
        std::cout << "[" << get_current_timestamp() << "] Input: " << test_input << std::endl;
        std::string response = model.generate_response(test_input);
        std::cout << "[" << get_current_timestamp() << "] Response: " << response << std::endl;
        
        std::cout << "[" << get_current_timestamp() << "] Conversation demo completed." << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "[" << get_current_timestamp() << "] Error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "[" << get_current_timestamp() << "] Unknown error occurred." << std::endl;
        return 1;
    }
    
    return 0;
}
