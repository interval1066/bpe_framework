#include "lm/tokenizer/bpe_tokenizer.hpp"
#include <iostream>
#include <vector>

int main() {
    lm::BPETokenizer tokenizer;
    
    // Training corpus with Unicode text
    std::vector<std::string> corpus = {
        "the quick brown fox jumps over the lazy dog",
        "artificial intelligence is transforming the world",
        "C++ is a powerful programming language",
        "machine learning models require large amounts of data",
        "你好世界", // Hello world in Chinese
        "こんにちは世界", // Hello world in Japanese
        "안녕하세요 세계", // Hello world in Korean
        "مرحبا بالعالم", // Hello world in Arabic
        "Γειά σου Κόσμε", // Hello world in Greek
        "Привет мир", // Hello world in Russian
        "नमस्ते दुनिया" // Hello world in Hindi
    };
    
    try {
        // Train the tokenizer
        std::cout << "Training tokenizer with Unicode text..." << std::endl;
        tokenizer.train(corpus, 1000);
        std::cout << "Vocabulary size: " << tokenizer.vocab_size() << std::endl;
        
        // Test encoding/decoding with various scripts
        std::vector<std::string> test_texts = {
            "hello world",
            "你好世界",
            "こんにちは世界",
            "مرحبا بالعالم",
            "Привет мир"
        };
        
        for (const auto& test_text : test_texts) {
            auto tokens = tokenizer.encode(test_text);
            std::string decoded = tokenizer.decode(tokens);
            
            std::cout << "\nOriginal: " << test_text << std::endl;
            std::cout << "Tokens: ";
            for (auto token : tokens) {
                std::cout << token << " ";
            }
            std::cout << std::endl;
            std::cout << "Decoded: " << decoded << std::endl;
            std::cout << "Match: " << (test_text == decoded ? "YES" : "NO") << std::endl;
        }
        
        // Save and load test
        tokenizer.save("unicode_bpe_model.txt");
        
        lm::BPETokenizer loaded_tokenizer;
        if (loaded_tokenizer.load("unicode_bpe_model.txt")) {
            std::cout << "\nSuccessfully loaded Unicode tokenizer" << std::endl;
            std::cout << "Loaded vocabulary size: " << loaded_tokenizer.vocab_size() << std::endl;
            
            // Test with the loaded tokenizer
            std::string test_text = "你好世界";
            auto tokens = loaded_tokenizer.encode(test_text);
            std::string decoded = loaded_tokenizer.decode(tokens);
            
            std::cout << "Loaded tokenizer test:" << std::endl;
            std::cout << "Original: " << test_text << std::endl;
            std::cout << "Decoded: " << decoded << std::endl;
            std::cout << "Match: " << (test_text == decoded ? "YES" : "NO") << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
