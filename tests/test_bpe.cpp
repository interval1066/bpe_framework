#include "lm/tokenizer/bpe_tokenizer.hpp"
#include <iostream>
#include <vector>

int main() {
    lm::BPETokenizer tokenizer;
    
    // Training corpus
    std::vector<std::string> corpus = {
        "the quick brown fox jumps over the lazy dog",
        "artificial intelligence is transforming the world",
        "C++ is a powerful programming language",
        "machine learning models require large amounts of data"
    };
    
    try {
        // Train the tokenizer
        std::cout << "Training tokenizer..." << std::endl;
        tokenizer.train(corpus, 500);
        std::cout << "Vocabulary size: " << tokenizer.vocab_size() << std::endl;
        
        // Test encoding/decoding
        std::string test_text = "the quick brown fox";
        auto tokens = tokenizer.encode(test_text);
        std::string decoded = tokenizer.decode(tokens);
        
        std::cout << "Original: " << test_text << std::endl;
        std::cout << "Tokens: ";
        for (auto token : tokens) {
            std::cout << token << " ";
        }
        std::cout << std::endl;
        std::cout << "Decoded: " << decoded << std::endl;
        
        // Save and load test
        tokenizer.save("bpe_model.txt");
        
        lm::BPETokenizer loaded_tokenizer;
        if (loaded_tokenizer.load("bpe_model.txt")) {
            std::cout << "Successfully loaded tokenizer" << std::endl;
            std::cout << "Loaded vocabulary size: " << loaded_tokenizer.vocab_size() << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

