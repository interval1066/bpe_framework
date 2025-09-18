// starter_demo.cpp
#include "lm/models/conversation_model.hpp"
#include "lm/tokenizer/bpe_tokenizer.hpp"
#include "lm/data/training_data.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>
#include <memory>
#include <algorithm>
#include <random>
#include <sstream>

// Helper function to get current timestamp
std::string get_current_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %X");
    return ss.str();
}

// Function to create comprehensive training dataset
lm::TrainingDataset create_training_dataset() {
    lm::TrainingDataset dataset;
    
    // Basic greetings
    dataset.add_example("Hello", "Hi there! How can I help you?");
    dataset.add_example("Hi", "Hello! What can I do for you?");
    dataset.add_example("Hey", "Hey! How can I assist you today?");
    dataset.add_example("Good morning", "Good morning! How are you today?");
    dataset.add_example("Good afternoon", "Good afternoon! How can I help you?");
    dataset.add_example("Good evening", "Good evening! What can I do for you?");
    
    // Questions about the AI
    dataset.add_example("What's your name?", "I'm an AI assistant created to help with conversations.");
    dataset.add_example("Who are you?", "I'm a conversational AI designed to chat with humans.");
    dataset.add_example("What can you do?", "I can chat, answer questions, and learn from our conversations.");
    dataset.add_example("Are you a robot?", "I'm an AI program, so in a way, yes! But I'm designed to be helpful and friendly.");
    dataset.add_example("Who created you?", "I was created by a developer interested in conversational AI.");
    
    // Common questions
    dataset.add_example("How are you?", "I'm functioning well, thank you for asking! How are you?");
    dataset.add_example("What's the weather like?", "I don't have access to real-time weather data, but I can chat about weather in general!");
    dataset.add_example("Tell me a joke", "Why don't scientists trust atoms? Because they make up everything!");
    dataset.add_example("What time is it?", "I don't have access to the current time, but I can help with other questions!");
    dataset.add_example("How old are you?", "I was just created, so I'm very new to the world!");
    
    // Jokes and fun responses
    dataset.add_example("Tell me another joke", "Why did the scarecrow win an award? Because he was outstanding in his field!");
    dataset.add_example("More jokes", "What do you call a fake noodle? An impasta!");
    dataset.add_example("That's funny", "I'm glad you liked it! Do you want to hear another one?");
    dataset.add_example("No thanks", "No problem! What would you like to talk about instead?");
    dataset.add_example("Yes please", "Great! What did the ocean say to the beach? Nothing, it just waved!");
    
    // Conversation continuations
    dataset.add_example("I'm good", "That's great to hear! How can I help you today?");
    dataset.add_example("I'm fine", "I'm glad to hear that! What would you like to talk about?");
    dataset.add_example("Not so good", "I'm sorry to hear that. Is there anything I can do to help?");
    dataset.add_example("What should we talk about?", "We could talk about technology, science, jokes, or anything you're interested in!");
    dataset.add_example("I don't know", "That's okay! We can just chat. How was your day?");
    
    // Multi-turn conversations
    std::vector<std::string> conversation1 = {
        "Hello",
        "Hi there! How can I help you?",
        "What's your name?",
        "I'm an AI assistant. What's yours?",
        "My name is Alex",
        "Nice to meet you, Alex! How can I help you today?"
    };
    dataset.add_conversation(conversation1);
    
    std::vector<std::string> conversation2 = {
        "Tell me a joke",
        "Why did the scarecrow win an award? Because he was outstanding in his field!",
        "That's pretty good",
        "Thanks! I have more if you'd like to hear them.",
        "Maybe later",
        "Sure thing! Just let me know when you're ready for more jokes."
    };
    dataset.add_conversation(conversation2);
    
    std::vector<std::string> conversation3 = {
        "How are you today?",
        "I'm doing well, thank you for asking! How about you?",
        "I'm good too",
        "That's great to hear! Is there anything I can help you with?",
        "Just wanted to chat",
        "I'd love to chat! What would you like to talk about?"
    };
    dataset.add_conversation(conversation3);
    
    return dataset;
}

int main() {
    std::cout << "[" << get_current_timestamp() << "] Starting conversation model initialization..." << std::endl;
    
    try {
        // Initialize tokenizer
        std::cout << "[" << get_current_timestamp() << "] Creating BPE tokenizer..." << std::endl;
        auto tokenizer = std::make_shared<lm::BPETokenizer>();
        
        // Create comprehensive tokenizer training data
        std::cout << "[" << get_current_timestamp() << "] Preparing training data for tokenizer..." << std::endl;
        std::vector<std::string> tokenizer_training_data = {
            "Hello, how are you?",
            "I'm doing well, thank you!",
            "What can I help you with today?",
            "The weather is nice today.",
            "I enjoy programming in C++.",
            "Machine learning is fascinating.",
            "Natural language processing enables computers to understand human language.",
            "This is a test of the tokenizer system.",
            "Reinforcement learning uses rewards to train agents.",
            "Deep learning models have many layers.",
            "The quick brown fox jumps over the lazy dog.",
            "Artificial intelligence is transforming the world.",
            "Machine learning enables computers to learn from data.",
            "Natural language processing helps computers understand human language.",
            "Conversational AI is becoming increasingly sophisticated.",
            "Transformers have revolutionized natural language processing.",
            "Attention mechanisms allow models to focus on relevant information.",
            "Tokenization is the first step in many NLP pipelines.",
            "Word embeddings represent words as vectors in high-dimensional space.",
            "Language models can generate human-like text."
        };
        
        // Add data from our conversation dataset
        auto conversation_dataset = create_training_dataset();
        for (const auto& example : conversation_dataset.examples()) {
            tokenizer_training_data.push_back(example.input);
            tokenizer_training_data.push_back(example.target);
        }
        
        std::cout << "[" << get_current_timestamp() << "] Training tokenizer with " 
                  << tokenizer_training_data.size() << " examples..." << std::endl;
        tokenizer->train(tokenizer_training_data, 1000);
        std::cout << "[" << get_current_timestamp() << "] Tokenizer training completed. Vocabulary size: " 
                  << tokenizer->vocab_size() << std::endl;
        
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
        model.set_verbose(true);     // Enable verbose output
        model.set_max_response_length(20); // Limit response length
        
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
        
        if (test_text != decoded_text) {
            std::cerr << "[" << get_current_timestamp() << "] ERROR: Tokenizer round-trip failed!" << std::endl;
            return 1;
        }
        
        // Create training dataset
        std::cout << "[" << get_current_timestamp() << "] Preparing conversation training data..." << std::endl;
        auto training_dataset = create_training_dataset();
        
        std::cout << "[" << get_current_timestamp() << "] Training conversation model with " 
                  << training_dataset.size() << " examples..." << std::endl;
        
        try {
            // Train for multiple epochs with learning rate decay
            const size_t num_epochs = 50;
            float learning_rate = 0.01f;
            
            for (size_t epoch = 0; epoch < num_epochs; epoch++) {
                std::cout << "[" << get_current_timestamp() << "] Epoch " << (epoch + 1) 
                          << "/" << num_epochs << " (LR: " << learning_rate << ")" << std::endl;
                
                // Shuffle the dataset for each epoch
                training_dataset.shuffle();
                
                // Train on the shuffled dataset
                model.train(training_dataset, 1, learning_rate); // Train for 1 epoch with current learning rate
                
                // Decay learning rate
                learning_rate *= 0.95f;
                
                // Save checkpoint every 10 epochs
                if ((epoch + 1) % 10 == 0) {
                    std::string checkpoint_name = "checkpoint_epoch_" + std::to_string(epoch + 1) + ".bin";
                    if (model.save_checkpoint(checkpoint_name)) {
                        std::cout << "[" << get_current_timestamp() << "] Saved checkpoint: " 
                                  << checkpoint_name << std::endl;
                    }
                }
            }
            
            // Save final model
            if (model.save_checkpoint("final_model.bin")) {
                std::cout << "[" << get_current_timestamp() << "] Saved final model." << std::endl;
            }
            
            std::cout << "[" << get_current_timestamp() << "] Model training completed." << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[" << get_current_timestamp() << "] Error during training: " << e.what() << std::endl;
            return 1;
        }
        
        // Test with various inputs
        std::cout << "[" << get_current_timestamp() << "] Testing model with sample inputs..." << std::endl;
        
        std::vector<std::string> test_inputs = {
            "Hello",
            "What's your name?",
            "Tell me a joke",
            "How are you?",
            "What can you do?"
        };
        
        for (const auto& test_input : test_inputs) {
            std::cout << "[" << get_current_timestamp() << "] Input: " << test_input << std::endl;
            std::string response = model.generate_response(test_input);
            std::cout << "[" << get_current_timestamp() << "] Response: " << response << std::endl;
            std::cout << "----------------------------------------" << std::endl;
        }
        
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
