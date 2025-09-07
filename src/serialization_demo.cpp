#include "lm/tokenizer/bpe_tokenizer.hpp"
#include "lm/optimizers/adam.hpp"
#include "lm/conversation_manager.hpp"
#include "lm/core/tensor.hpp"
#include <iostream>
#include <fstream>
#include <chrono>

using namespace lm;

int main() {
    std::cout << "=== BPE Framework Serialization Demo ===\n\n";
    
    try {
        // Initialize tokenizer
        BPETokenizer tokenizer;
        
        // Create a small test corpus
        std::vector<std::string> corpus = {
            "The quick brown fox jumps over the lazy dog",
            "Programming is fun with C++ and machine learning",
            "Natural language processing transforms how we interact with computers"
        };
        
        std::cout << "Training tokenizer on " << corpus.size() << " sentences...\n";
        tokenizer.train(corpus, 100); // Small vocabulary for testing
        
        // Test conversation manager
        std::cout << "Testing conversation manager...\n";
        ConversationManager conv_manager;
        
        // Create a conversation and add some messages
        std::string conv_id = conv_manager.create_conversation("Test Conversation");
        conv_manager.add_message(conv_id, "user", "Hello, how are you?");
        conv_manager.add_message(conv_id, "assistant", "I'm doing well, thank you!");
        conv_manager.add_message(conv_id, "user", "What's the weather like today?");
        
        // Save conversation
        std::cout << "Saving conversation...\n";
        conv_manager.save_conversations("test_conversations.bin");
        
        // Load conversation into a new manager
        std::cout << "Loading conversation...\n";
        ConversationManager loaded_conv_manager;
        loaded_conv_manager.load_conversations("test_conversations.bin");
        
        // Verify the loaded conversation
        auto loaded_conv = loaded_conv_manager.get_conversation(conv_id);
        if (loaded_conv) {
            std::cout << "Loaded conversation has " << loaded_conv->turns.size() << " turns\n";
            for (size_t i = 0; i < loaded_conv->turns.size(); i++) {
                const auto& turn = loaded_conv->turns[i];
                std::cout << "Turn " << i << ": " << speaker_type_to_string(turn.speaker) 
                          << ": " << turn.text << "\n";
            }
        }
        
        // Test optimizer state serialization
        std::cout << "Testing optimizer state serialization...\n";
        
        // Create a simple set of parameters for the optimizer
        std::vector<Tensor> params;
        params.push_back(Tensor({2, 3}, true)); // parameter with requires_grad = true
        params.push_back(Tensor({5}, true));    // another parameter
        
        // Initialize an optimizer
        AdamOptimizer optimizer(0.001, 0.9, 0.999, 1e-8);
        
        // Initialize moments for the parameters
        optimizer.initialize_moments(params);
        
        // Save optimizer state
        optimizer.save_state("test_optimizer.bin");
        
        // Create a new optimizer and load the state
        AdamOptimizer new_optimizer(0.001, 0.9, 0.999, 1e-8);
        new_optimizer.load_state("test_optimizer.bin");
        std::cout << "Optimizer state loaded successfully\n";
        
        // Test tensor serialization
        std::cout << "Testing tensor serialization...\n";
        
        // Create a tensor with explicit shape vector to avoid ambiguity
        std::vector<size_t> shape = {2, 3};
        Tensor test_tensor(shape);
        test_tensor.data() << 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f;
        
        {
            std::ofstream ofs("test_tensor.bin", std::ios::binary);
            cereal::BinaryOutputArchive archive(ofs);
            archive(test_tensor);
        }
        
        Tensor loaded_tensor;
        {
            std::ifstream ifs("test_tensor.bin", std::ios::binary);
            cereal::BinaryInputArchive archive(ifs);
            archive(loaded_tensor);
        }
        
        std::cout << "Original tensor:\n" << test_tensor.data() << "\n";
        std::cout << "Loaded tensor:\n" << loaded_tensor.data() << "\n";
        
        // Test tokenizer serialization (if implemented)
        std::cout << "Testing tokenizer serialization...\n";
        tokenizer.save("test_tokenizer.bin");
        
        BPETokenizer loaded_tokenizer;
        loaded_tokenizer.load("test_tokenizer.bin");
        std::cout << "Tokenizer vocabulary size after loading: " << loaded_tokenizer.vocab_size() << "\n";
        
        std::cout << "\n=== Serialization Demo Completed Successfully ===\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}

