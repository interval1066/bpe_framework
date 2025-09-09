#include <iostream>
#include "lm/models/transformer_model.hpp"  // Use the correct header

int main() {
    // Use TransformerModel instead of Transformer
    lm::TransformerModel model(1000, 512, 6, 8, 2048, 0.1f);
    
    std::cout << "Transformer model created successfully!" << std::endl;
    std::cout << "Vocabulary size: " << model.get_vocab_size() << std::endl;
    std::cout << "Model dimensions: " << model.get_d_model() << std::endl;
    
    // Test with some sample tokens
    std::vector<lm::TokenID> test_tokens = {1, 2, 3, 4, 5};
    
    try {
        auto output = model.forward(test_tokens);
        std::cout << "Forward pass completed successfully!" << std::endl;
        std::cout << "Output size: " << output.size() << std::endl;
        
        // Test generation
        auto generated = model.generate(test_tokens, 10, 0.8f);
        std::cout << "Generated tokens: ";
        for (auto token : generated) {
            std::cout << token << " ";
        }
        std::cout << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error during forward pass: " << e.what() << std::endl;
    }
    
    return 0;
}

