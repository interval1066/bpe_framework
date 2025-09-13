// src/test_data_loader.cpp
#include <lm/training/data_loader.hpp>
#include <lm/training/losses.hpp>
#include <lm/tokenizer/bpe_tokenizer.hpp>
#include <iostream>

int main() {
    // Create a simple tokenizer for testing
    lm::BPETokenizer tokenizer;
    // Initialize with a small vocabulary for testing
    // (You'll need to implement a way to create a test tokenizer)
    
    try {
        // Create data loader
        lm::ConversationDataLoader loader("test_conversations.txt", tokenizer, 2, 10);
        
        std::cout << "Number of batches: " << loader.num_batches() << std::endl;
        
        while (loader.has_next()) {
            auto [inputs, targets] = loader.next_batch();
            std::cout << "Input shape: [";
            for (auto dim : inputs.shape()) std::cout << dim << ", ";
            std::cout << "], Target shape: [";
            for (auto dim : targets.shape()) std::cout << dim << ", ";
            std::cout << "]" << std::endl;
        }
        
        std::cout << "Data loader test completed successfully!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

