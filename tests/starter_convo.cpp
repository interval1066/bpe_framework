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
#include <fstream>
#include <string>

// Helper function to get current timestamp
std::string get_current_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %X");
    return ss.str();
}

// Function to load training data from a text file
lm::TrainingDataset load_training_dataset(const std::string& filename) {
    lm::TrainingDataset dataset;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "[" << get_current_timestamp() << "] ERROR: Could not open training data file: " << filename << std::endl;
        return dataset;
    }
    
    std::string line;
    std::vector<std::string> current_conversation;
    
    while (std::getline(file, line)) {
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        // Check if this line starts a new conversation
        if (line.find("CONVERSATION:") == 0) {
            if (!current_conversation.empty()) {
                dataset.add_conversation(current_conversation);
                current_conversation.clear();
            }
            continue;
        }
        
        // Check if this is a single example (input and response separated by tab)
        size_t tab_pos = line.find('\t');
        if (tab_pos != std::string::npos) {
            std::string input = line.substr(0, tab_pos);
            std::string target = line.substr(tab_pos + 1);
            dataset.add_example(input, target);
        } else {
            // Add line to current conversation
            current_conversation.push_back(line);
        }
    }
    
    // Add the last conversation if exists
    if (!current_conversation.empty()) {
        dataset.add_conversation(current_conversation);
    }
    
    file.close();
    return dataset;
}

int main() {
    std::cout << "[" << get_current_timestamp() << "] Starting conversation model initialization..." << std::endl;
    
    try {
        // Initialize tokenizer
        std::cout << "[" << get_current_timestamp() << "] Creating BPE tokenizer..." << std::endl;
        auto tokenizer = std::make_shared<lm::BPETokenizer>();
        
        // Load training data from file
        std::cout << "[" << get_current_timestamp() << "] Loading training data from file..." << std::endl;
        auto training_dataset = load_training_dataset("training_data.txt");
        
        if (training_dataset.size() == 0) {
            std::cerr << "[" << get_current_timestamp() << "] ERROR: No training data loaded!" << std::endl;
            return 1;
        }
        
        std::cout << "[" << get_current_timestamp() << "] Loaded " << training_dataset.size() << " training examples." << std::endl;
        
        // Create comprehensive tokenizer training data
        std::cout << "[" << get_current_timestamp() << "] Preparing training data for tokenizer..." << std::endl;
        std::vector<std::string> tokenizer_training_data;
        
        // Add general text data for tokenizer
        std::vector<std::string> general_texts = {
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
        
        for (const auto& text : general_texts) {
            tokenizer_training_data.push_back(text);
        }
        
        // Add data from our conversation dataset
        for (const auto& example : training_dataset.examples()) {
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
