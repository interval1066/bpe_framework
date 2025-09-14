// src/serialization_demo.cpp
#include "lm/tokenizer/bpe_tokenizer.hpp"
#include "lm/optimizers/adam.hpp"
#include "lm/conversation/conversation_manager.hpp"
#include "lm/core/tensor.hpp"
#include "lm/generation/greedy_sampler.hpp"
#include <iostream>
#include <fstream>
#include <chrono>
#include <memory>

using namespace lm;

int main() {
    std::cout << "=== BPE Framework Serialization Demo ===\n\n";
    
    try {
        // Create tokenizer directly as a shared_ptr
        auto tokenizer_ptr = std::make_shared<BPETokenizer>();
        
        // Create a small test corpus
        std::vector<std::string> corpus = {
            "The quick brown fox jumps over the lazy dog",
            "Programming is fun with C++ and machine learning",
            "Natural language processing transforms how we interact with computers"
        };
        
        std::cout << "Training tokenizer on " << corpus.size() << " sentences...\n";
        tokenizer_ptr->train(corpus, 100); // Small vocabulary for testing
        
        // Create a dummy model for the conversation manager
        auto dummy_model = std::make_shared<TransformerModel>(1000, 512, 6, 8, 2048, 0.1f);
        auto dummy_sampler = std::make_unique<GreedySampler>();
        
        // Test conversation manager
        std::cout << "Testing conversation manager...\n";
        ConversationManager conv_manager(dummy_model, tokenizer_ptr, std::move(dummy_sampler));
        
        // Add some messages to the conversation
        conv_manager.add_to_history("Hello, how are you?", true);
        conv_manager.add_to_history("I'm doing well, thank you!", false);
        conv_manager.add_to_history("What's the weather like today?", true);
        
        // Save conversation
        std::cout << "Saving conversation...\n";
        conv_manager.save_conversation("test_conversation.txt");
        
        // Load conversation into a new manager
        std::cout << "Loading conversation...\n";
        auto dummy_model2 = std::make_shared<TransformerModel>(1000, 512, 6, 8, 2048, 0.1f);
        auto dummy_sampler2 = std::make_unique<GreedySampler>();
        ConversationManager loaded_conv_manager(dummy_model2, tokenizer_ptr, std::move(dummy_sampler2));
        loaded_conv_manager.load_conversation("test_conversation.txt");
        
        // Verify the loaded conversation
        auto history = loaded_conv_manager.get_history();
        std::cout << "Loaded conversation has " << history.size() << " turns\n";
        for (size_t i = 0; i < history.size(); i++) {
            std::cout << "Turn " << i << ": " << history[i] << "\n";
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
        
        // Test tokenizer serialization
        std::cout << "Testing tokenizer serialization...\n";
        tokenizer_ptr->save("test_tokenizer.bin");
        
        // Create a new tokenizer and load the saved state
        auto loaded_tokenizer_ptr = std::make_shared<BPETokenizer>();
        loaded_tokenizer_ptr->load("test_tokenizer.bin");
        std::cout << "Tokenizer vocabulary size after loading: " << loaded_tokenizer_ptr->vocab_size() << "\n";
        
        std::cout << "\n=== Serialization Demo Completed Successfully ===\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}

