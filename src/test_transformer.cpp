// src/test_transformer.cpp
#include "lm/models/transformer_model.hpp"
#include <iostream>
#include <memory>

using namespace lm;

int main() {
    try {
        // Create a transformer model
        auto model = TransformerModel::create(10000, 512, 6, 8, 2048, 0.1f);
        
        std::cout << "Transformer model created successfully!" << std::endl;
        
        // Test forward pass with some dummy input
        std::vector<TokenID> input_tokens = {1, 2, 3, 4, 5};
        
        std::cout << "Running forward pass..." << std::endl;
        auto logits = model->forward(input_tokens);
        
        std::cout << "Forward pass completed successfully!" << std::endl;
        std::cout << "Output logits size: " << logits.size() << std::endl;
        
        // Test generation
        std::cout << "Testing generation..." << std::endl;
        auto generated = model->generate(input_tokens, 10, 0.8f);
        
        std::cout << "Generated sequence: ";
        for (const auto& token : generated) {
            std::cout << token << " ";
        }
        std::cout << std::endl;
        
        // Test serialization
        std::cout << "Testing serialization..." << std::endl;
        bool saved = model->save("test_model.bin");
        
        if (saved) {
            std::cout << "Model saved successfully!" << std::endl;
            
            // Load the model
            auto loaded_model = TransformerModel::create();
            bool loaded = loaded_model->load("test_model.bin");
            
            if (loaded) {
                std::cout << "Model loaded successfully!" << std::endl;
                
                // Test the loaded model
                auto loaded_logits = loaded_model->forward(input_tokens);
                std::cout << "Loaded model output size: " << loaded_logits.size() << std::endl;
            } else {
                std::cout << "Failed to load model!" << std::endl;
            }
        } else {
            std::cout << "Failed to save model!" << std::endl;
        }
        
        std::cout << "All tests completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

