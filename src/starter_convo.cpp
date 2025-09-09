// main.cpp
#include "lm/models/conversation_model.hpp"
#include "lm/tokenizer/bpe_tokenizer.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>

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
    tokenizer->train(training_data, 1000);  // Reduced vocab size for demo
    std::cout << "[" << get_current_timestamp() << "] Tokenizer training completed. Vocabulary size: " << tokenizer->vocab_size() << std::endl;
    
    // Initialize conversation model
    std::cout << "[" << get_current_timestamp() << "] Initializing conversation model..." << std::endl;
    lm::ConversationModel model(tokenizer->vocab_size());
    model.set_tokenizer(tokenizer);
    
    // Train the model
    std::cout << "[" << get_current_timestamp() << "] Preparing conversation training data..." << std::endl;
    std::vector<std::string> conversations = {
        "<|user|>Hello<|endoftext|><|assistant|>Hi there! How can I help you?<|endoftext|>",
        "<|user|>What's the weather like?<|endoftext|><|assistant|>I'm not sure, I don't have access to real-time weather data.<|endoftext|>",
        "<|user|>What can you do?<|endoftext|><|assistant|>I can chat with you about various topics and answer questions based on my training.<|endoftext|>",
        "<|user|>Tell me a joke<|endoftext|><|assistant|>Why don't scientists trust atoms? Because they make up everything!<|endoftext|>",
        "<|user|>How does machine learning work?<|endoftext|><|assistant|>Machine learning uses algorithms to learn patterns from data without being explicitly programmed for each task.<|endoftext|>"
    };
    
    std::cout << "[" << get_current_timestamp() << "] Training conversation model with " << conversations.size() << " examples..." << std::endl;
    model.train(conversations);
    std::cout << "[" << get_current_timestamp() << "] Model training completed." << std::endl;
    
    // Test with some sample inputs
    std::cout << "[" << get_current_timestamp() << "] Testing model with sample inputs..." << std::endl;
    std::vector<std::string> test_inputs = {
        "Hello, how are you?",
        "What can you do?",
        "Tell me about machine learning"
    };
    
    for (const auto& input : test_inputs) {
        std::cout << "[" << get_current_timestamp() << "] Input: " << input << std::endl;
        std::string response = model.generate_response(input);
        std::cout << "[" << get_current_timestamp() << "] Response: " << response << std::endl;
        std::cout << "[" << get_current_timestamp() << "] ---" << std::endl;
    }
    
    // Interactive conversation loop
    std::cout << "[" << get_current_timestamp() << "] Starting interactive conversation mode..." << std::endl;
    std::cout << "[" << get_current_timestamp() << "] Type 'quit' to exit, 'clear' to reset conversation context" << std::endl;
    
    std::string user_input;
    while (true) {
        std::cout << "[" << get_current_timestamp() << "] User: ";
        std::getline(std::cin, user_input);
        
        if (user_input == "quit" || user_input == "exit") {
            break;
        }
        
        if (user_input == "clear") {
            // Assuming there's a method to clear context
            // model.clear_context();
            std::cout << "[" << get_current_timestamp() << "] Conversation context cleared." << std::endl;
            continue;
        }
        
        if (user_input.empty()) {
            continue;
        }
        
        try {
            std::string response = model.generate_response(user_input);
            std::cout << "[" << get_current_timestamp() << "] AI: " << response << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[" << get_current_timestamp() << "] Error generating response: " << e.what() << std::endl;
        }
    }
    
    // Save the model
    std::cout << "[" << get_current_timestamp() << "] Saving model to 'conversation_model.bin'..." << std::endl;
    model.save_model("conversation_model.bin");
    std::cout << "[" << get_current_timestamp() << "] Model saved successfully." << std::endl;
    
    std::cout << "[" << get_current_timestamp() << "] Conversation demo completed." << std::endl;
    return 0;
}

