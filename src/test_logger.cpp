#include "lm/tokenizer/bpe_tokenizer.hpp"
#include <iostream>
#include <vector>
#include <string>

using namespace lm;

void run_basic_test() {
    std::cout << "=== BASIC TEST ===" << std::endl;
    
    BPETokenizer tokenizer;
    tokenizer.enable_debug_logging(true);
    
    // Train on a simple corpus
    std::vector<std::string> corpus = {
        "The quick brown fox jumps over the lazy dog.",
        "I love machine learning and natural language processing!",
        "Byte Pair Encoding is an effective tokenization method."
    };
    
    std::cout << "Training tokenizer..." << std::endl;
    tokenizer.train(corpus, 300);
    std::cout << "Training completed. Vocabulary size: " << tokenizer.vocab_size() << std::endl;
    
    // Test encoding and decoding
    std::string test_text = "The quick brown fox";
    std::cout << "\nTesting encoding/decoding with: '" << test_text << "'" << std::endl;
    
    auto tokens = tokenizer.encode(test_text);
    std::string decoded = tokenizer.decode(tokens);
    
    std::cout << "\nOriginal: '" << test_text << "'" << std::endl;
    std::cout << "Decoded:  '" << decoded << "'" << std::endl;
    std::cout << "Tokens: [";
    for (size_t i = 0; i < tokens.size(); i++) {
        std::cout << tokens[i];
        if (i < tokens.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    // Dump vocabulary and merges for inspection
    std::cout << "\nVocabulary:" << std::endl;
    tokenizer.dump_vocabulary();
    
    std::cout << "\nMerges:" << std::endl;
    tokenizer.dump_merges();
}

void run_unicode_test() {
    std::cout << "\n\n=== UNICODE TEST ===" << std::endl;
    
    BPETokenizer tokenizer;
    tokenizer.enable_debug_logging(true);
    
    // Train on a corpus with Unicode characters
    std::vector<std::string> corpus = {
        "Hello world! 你好世界!",
        "Bonjour le monde! ¡Hola mundo!",
        "Café résumé naïve façade",
        "Emoji: 😊 🚀 🌟 🎉"
    };
    
    std::cout << "Training tokenizer with Unicode..." << std::endl;
    tokenizer.train(corpus, 400);
    std::cout << "Training completed. Vocabulary size: " << tokenizer.vocab_size() << std::endl;
    
    // Test encoding and decoding with Unicode
    std::string test_text = "Café résumé with emoji 😊";
    std::cout << "\nTesting encoding/decoding with: '" << test_text << "'" << std::endl;
    
    auto tokens = tokenizer.encode(test_text);
    std::string decoded = tokenizer.decode(tokens);
    
    std::cout << "\nOriginal: '" << test_text << "'" << std::endl;
    std::cout << "Decoded:  '" << decoded << "'" << std::endl;
    std::cout << "Tokens: [";
    for (size_t i = 0; i < tokens.size(); i++) {
        std::cout << tokens[i];
        if (i < tokens.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

void run_edge_case_test() {
    std::cout << "\n\n=== EDGE CASE TEST ===" << std::endl;
    
    BPETokenizer tokenizer;
    tokenizer.enable_debug_logging(true);
    
    // Train on a small corpus
    std::vector<std::string> corpus = {
        "a b c d e f g h i j k l m n o p q r s t u v w x y z",
        "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z",
        "0 1 2 3 4 5 6 7 8 9",
        "! @ # $ % ^ & * ( ) - _ = + [ ] { } ; : ' \" , . < > / ?"
    };
    
    std::cout << "Training tokenizer with edge cases..." << std::endl;
    tokenizer.train(corpus, 200);
    std::cout << "Training completed. Vocabulary size: " << tokenizer.vocab_size() << std::endl;
    
    // Test various edge cases
    std::vector<std::string> test_cases = {
        "a",
        "abc",
        "hello world",
        "!@#$%",
        "a b c",
        "The quick brown fox"
    };
    
    for (const auto& test_text : test_cases) {
        std::cout << "\nTesting: '" << test_text << "'" << std::endl;
        
        auto tokens = tokenizer.encode(test_text);
        std::string decoded = tokenizer.decode(tokens);
        
        std::cout << "Original: '" << test_text << "'" << std::endl;
        std::cout << "Decoded:  '" << decoded << "'" << std::endl;
        std::cout << "Match: " << (test_text == decoded ? "YES" : "NO") << std::endl;
        std::cout << "Tokens: [";
        for (size_t i = 0; i < tokens.size(); i++) {
            std::cout << tokens[i];
            if (i < tokens.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
}

void run_save_load_test() {
    std::cout << "\n\n=== SAVE/LOAD TEST ===" << std::endl;
    
    BPETokenizer tokenizer;
    
    // Train on a simple corpus
    std::vector<std::string> corpus = {
        "The quick brown fox jumps over the lazy dog.",
        "I love programming in C++",
        "Machine learning is fascinating"
    };
    
    std::cout << "Training tokenizer..." << std::endl;
    tokenizer.train(corpus, 250);
    std::cout << "Training completed. Vocabulary size: " << tokenizer.vocab_size() << std::endl;
    
    // Test encoding before save
    std::string test_text = "quick brown fox";
    auto original_tokens = tokenizer.encode(test_text);
    std::string original_decoded = tokenizer.decode(original_tokens);
    
    std::cout << "Before save - Original: '" << test_text << "'" << std::endl;
    std::cout << "Before save - Decoded:  '" << original_decoded << "'" << std::endl;
    
    // Save the tokenizer
    std::string filename = "bpe_tokenizer.model";
    if (tokenizer.save(filename)) {
        std::cout << "Tokenizer saved to " << filename << std::endl;
    } else {
        std::cout << "Failed to save tokenizer to " << filename << std::endl;
        return;
    }
    
    // Load into a new tokenizer
    BPETokenizer loaded_tokenizer;
    if (loaded_tokenizer.load(filename)) {
        std::cout << "Tokenizer loaded from " << filename << std::endl;
        std::cout << "Loaded vocabulary size: " << loaded_tokenizer.vocab_size() << std::endl;
        
        // Test encoding after load
        auto loaded_tokens = loaded_tokenizer.encode(test_text);
        std::string loaded_decoded = loaded_tokenizer.decode(loaded_tokens);
        
        std::cout << "After load - Original: '" << test_text << "'" << std::endl;
        std::cout << "After load - Decoded:  '" << loaded_decoded << "'" << std::endl;
        std::cout << "Match: " << (original_decoded == loaded_decoded ? "YES" : "NO") << std::endl;
        
        // Compare tokens
        std::cout << "Original tokens: [";
        for (size_t i = 0; i < original_tokens.size(); i++) {
            std::cout << original_tokens[i];
            if (i < original_tokens.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        std::cout << "Loaded tokens: [";
        for (size_t i = 0; i < loaded_tokens.size(); i++) {
            std::cout << loaded_tokens[i];
            if (i < loaded_tokens.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    } else {
        std::cout << "Failed to load tokenizer from " << filename << std::endl;
    }
}

int main() {
    std::cout << "BPETokenizer Test Application" << std::endl;
    std::cout << "============================" << std::endl;
    
    try {
        run_basic_test();
        run_unicode_test();
        run_edge_case_test();
        run_save_load_test();
        
        std::cout << "\nAll tests completed!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
