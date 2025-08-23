#include "lm/tokenizer/bpe_tokenizer.hpp"
#include "lm/models/transformer.hpp"
#include "lm/core/tensor.hpp"
#include <iostream>
#include <vector>
#include <memory>
int main() {
    std::cout << "=== BPE Tokenizer and Transformer Integration Example ===\n";
    
    try {
        // Initialize BPE tokenizer
        lm::BPETokenizer tokenizer;
        
        // Sample training corpus
        std::vector<std::string> training_corpus = {
            "The quick brown fox jumps over the lazy dog",
            "Artificial intelligence is transforming the world",
            "Machine learning models require large amounts of data",
            "Natural language processing enables computers to understand human language",
            "Deep learning has revolutionized many fields of AI"
        };
        
        // Train the tokenizer
        std::cout << "Training BPE tokenizer...\n";
        tokenizer.train(training_corpus, 500);
        std::cout << "Tokenizer trained with vocabulary size: " << tokenizer.vocab_size() << "\n";
        
        // Test encoding and decoding
        std::string test_text = "The quick brown fox jumps over the lazy dog";
        std::cout << "\nOriginal text: " << test_text << "\n";
        
        // Encode text to token IDs
        auto token_ids = tokenizer.encode(test_text);
        std::cout << "Encoded token IDs: ";
        for (auto id : token_ids) {
            std::cout << id << " ";
        }
        std::cout << "\n";
        
        // Decode back to text
        std::string decoded_text = tokenizer.decode(token_ids);
        std::cout << "Decoded text: " << decoded_text << "\n";
        
        // Test Eigen integration
        std::cout << "\n=== Eigen Integration Test ===\n";
        Eigen::VectorXi eigen_tokens = tokenizer.encode_to_vector(test_text);
        std::cout << "Eigen vector size: " << eigen_tokens.size() << "\n";
        std::cout << "Eigen vector contents: " << eigen_tokens.transpose() << "\n";
        
        // Decode from Eigen vector
        std::string from_eigen = tokenizer.decode_from_vector(eigen_tokens);
        std::cout << "Text from Eigen vector: " << from_eigen << "\n";
        
        // Test token frequencies (placeholder implementation)
        auto frequencies = tokenizer.token_frequencies();
        std::cout << "Token frequencies vector size: " << frequencies.size() << "\n";
        
        // Initialize transformer model
        std::cout << "\n=== Transformer Model Test ===\n";
        size_t vocab_size = tokenizer.vocab_size();
        size_t d_model = 512;
        size_t num_heads = 8;
        size_t d_ff = 2048;
        size_t num_layers = 6;
        size_t max_seq_len = 512;
        
        lm::Transformer transformer(vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len);
        std::cout << "Transformer model initialized successfully\n";
        std::cout << "Model parameters: " << transformer.parameters().size() << " parameter tensors\n";
        
        // Prepare input for transformer (convert token IDs to tensor)
        if (!token_ids.empty()) {
            // Create a batch of size 1 with our token IDs
            lm::Tensor input_tensor({1, static_cast<size_t>(token_ids.size())});
            for (size_t i = 0; i < token_ids.size(); ++i) {
                input_tensor.data()(0, i) = static_cast<float>(token_ids[i]);
            }
            
            std::cout << "Input tensor shape: (" << input_tensor.shape()[0] 
                      << ", " << input_tensor.shape()[1] << ")\n";
            
            // Set model to evaluation mode
            transformer.set_training(false);
            
            // Forward pass (this would normally produce logits)
            try {
                lm::Tensor output = transformer(input_tensor);
                std::cout << "Transformer forward pass completed successfully\n";
                std::cout << "Output tensor shape: (" << output.shape()[0] 
                          << ", " << output.shape()[1] << ", " << output.shape()[2] << ")\n";
                
                // The output would be logits for next token prediction
                // In a real application, you would sample from these logits
            } catch (const std::exception& e) {
                std::cout << "Transformer forward pass failed: " << e.what() << "\n";
                std::cout << "This is expected if the transformer implementation is not complete yet\n";
            }
        }
        
        // Test serialization
        std::cout << "\n=== Serialization Test ===\n";
        bool save_success = tokenizer.save("test_tokenizer.bpe");
        if (save_success) {
            std::cout << "Tokenizer saved successfully\n";
            
            // Load into a new tokenizer
            lm::BPETokenizer loaded_tokenizer;
            bool load_success = loaded_tokenizer.load("test_tokenizer.bpe");
            if (load_success) {
                std::cout << "Tokenizer loaded successfully\n";
                
                // Test the loaded tokenizer
                std::string test_loaded = "Artificial intelligence";
                auto loaded_ids = loaded_tokenizer.encode(test_loaded);
                std::string loaded_decoded = loaded_tokenizer.decode(loaded_ids);
                std::cout << "Loaded tokenizer test: " << test_loaded << " -> " << loaded_decoded << "\n";
            } else {
                std::cout << "Failed to load tokenizer\n";
            }
            
            // Clean up
            remove("test_tokenizer.bpe");
        } else {
            std::cout << "Failed to save tokenizer\n";
        }
        
        std::cout << "\n=== Integration Example Completed Successfully ===\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}

