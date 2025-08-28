// examples/serialization_demo.cpp
#include "../include/lm/training/trainer.hpp"
#include "../include/lm/tokenizer/bpe_tokenizer.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

int main() {
    std::cout << "BPE Framework Serialization Demo" << std::endl;
    std::cout << "=================================" << std::endl;
    
    try {
        // Create a simple corpus for training
        std::vector<std::string> corpus = {
            "The quick brown fox jumps over the lazy dog",
            "Artificial intelligence is transforming our world",
            "Language models can understand and generate text",
            "Deep learning requires large amounts of data",
            "Natural language processing is a fascinating field"
        };
        
        // Initialize tokenizer
        lm::BPETokenizer tokenizer;
        tokenizer.train(corpus, 1000); // Train with 1000 merges
        
        // Initialize model and trainer
        size_t embedding_dim = 64;
        size_t hidden_dim = 128;
        size_t num_layers = 2;
        
        lm::LanguageModelTrainer trainer(tokenizer, embedding_dim, hidden_dim, num_layers);
        
        std::cout << "Model created with " << trainer.model().parameters().size() 
                  << " parameters" << std::endl;
        
        // Save the model before training
        std::string initial_model_path = "initial_model.bin";
        std::cout << "Saving initial model to " << initial_model_path << "..." << std::endl;
        trainer.save_model(initial_model_path);
        
        // Train for a few epochs (just for demonstration)
        std::cout << "Training model for 3 epochs..." << std::endl;
        trainer.train(corpus, 3, 2, 10); // 3 epochs, batch size 2, sequence length 10
        
        // Save the trained model
        std::string trained_model_path = "trained_model.bin";
        std::cout << "Saving trained model to " << trained_model_path << "..." << std::endl;
        trainer.save_model(trained_model_path);
        
        // Create a new trainer with the same architecture
        lm::LanguageModelTrainer new_trainer(tokenizer, embedding_dim, hidden_dim, num_layers);
        
        // Load the trained model
        std::cout << "Loading trained model from " << trained_model_path << "..." << std::endl;
        new_trainer.load_model(trained_model_path);
        
        // Verify that the parameters are the same
        auto original_params = trainer.model().parameters();
        auto loaded_params = new_trainer.model().parameters();
        
        std::cout << "Verifying parameter consistency..." << std::endl;
        
        if (original_params.size() != loaded_params.size()) {
            std::cerr << "Error: Parameter count mismatch!" << std::endl;
            return 1;
        }
        
        // Check if parameters are approximately equal (allowing for floating point precision)
        bool all_match = true;
        float max_diff = 0.0f;
        
        for (size_t i = 0; i < original_params.size(); ++i) {
            const auto& original = original_params[i].data();
            const auto& loaded = loaded_params[i].data();
            
            if (original.rows() != loaded.rows() || original.cols() != loaded.cols()) {
                std::cerr << "Error: Parameter shape mismatch at index " << i << std::endl;
                all_match = false;
                continue;
            }
            
            // Check if values are approximately equal
            for (int j = 0; j < original.size(); ++j) {
                float diff = std::abs(original(j) - loaded(j));
                if (diff > max_diff) {
                    max_diff = diff;
                }
                
                // Allow for small floating point differences
                if (diff > 1e-6) {
                    std::cerr << "Error: Parameter value mismatch at index " << i 
                              << ", position " << j << ": " << original(j) 
                              << " vs " << loaded(j) << std::endl;
                    all_match = false;
                }
            }
        }
        
        std::cout << "Maximum parameter difference: " << max_diff << std::endl;
        
        if (all_match) {
            std::cout << "SUCCESS: All parameters match correctly!" << std::endl;
        } else {
            std::cout << "WARNING: Some parameters don't match exactly" << std::endl;
        }
        
        // Test text generation with the loaded model
        std::cout << "\nTesting text generation with loaded model..." << std::endl;
        
        // Set model to evaluation mode
        new_trainer.model().eval();
        
        // Generate some text
        std::string prompt = "Artificial intelligence";
        std::cout << "Prompt: " << prompt << std::endl;
        
        // Tokenize the prompt
        auto tokens = tokenizer.encode(prompt);
        
        // Convert to tensor
        lm::Tensor input_tensor({static_cast<size_t>(tokens.size())});
        for (size_t i = 0; i < tokens.size(); ++i) {
            input_tensor(i) = static_cast<float>(tokens[i]);
        }
        
        // Forward pass (just for demonstration)
        auto output = new_trainer.model().forward(input_tensor);
        std::cout << "Model output shape: [";
        for (auto dim : output.shape()) {
            std::cout << dim << " ";
        }
        std::cout << "]" << std::endl;
        
        std::cout << "\nSerialization demo completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
