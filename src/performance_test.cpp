#include "lm/tokenizer/bpe_tokenizer.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <random>
#include <algorithm>
#include <sstream>  // Add this include for std::istringstream

// Generate random text for testing
std::vector<std::string> generate_test_corpus(size_t num_sentences, size_t min_words, size_t max_words) {
    std::vector<std::string> common_words = {
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
        "artificial", "intelligence", "machine", "learning", "deep", "neural", "network",
        "language", "model", "transformer", "attention", "mechanism", "tokenization",
        "byte", "pair", "encoding", "subword", "vocabulary", "training", "inference"
    };
    
    std::vector<std::string> corpus;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> word_count_dist(min_words, max_words);
    std::uniform_int_distribution<> word_index_dist(0, common_words.size() - 1);
    
    for (size_t i = 0; i < num_sentences; ++i) {
        int word_count = word_count_dist(gen);
        std::string sentence;
        
        for (int j = 0; j < word_count; ++j) {
            if (!sentence.empty()) {
                sentence += " ";
            }
            sentence += common_words[word_index_dist(gen)];
        }
        
        corpus.push_back(sentence);
    }
    
    return corpus;
}

// Measure memory usage (Linux specific)
size_t get_peak_memory_usage() {
    #ifdef __linux__
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.compare(0, 6, "VmPeak") == 0) {
            std::istringstream iss(line);
            std::string key;
            size_t value;
            std::string unit;
            iss >> key >> value >> unit;
            if (unit == "kB") {
                return value * 1024; // Convert to bytes
            }
        }
    }
    #endif
    return 0;
}

void run_performance_test() {
    std::cout << "=== BPE Tokenizer Performance Test ===\n";
    
    // Test different corpus sizes
    std::vector<size_t> corpus_sizes = {100, 1000, 5000};
    std::vector<size_t> vocab_sizes = {500, 1000, 2000};
    
    for (size_t corpus_size : corpus_sizes) {
        for (size_t vocab_size : vocab_sizes) {
            std::cout << "\n--- Test Configuration: " << corpus_size 
                      << " sentences, " << vocab_size << " vocabulary ---\n";
            
            // Generate test corpus
            auto corpus = generate_test_corpus(corpus_size, 5, 15);
            
            // Measure training performance
            auto start_time = std::chrono::high_resolution_clock::now();
            size_t start_memory = get_peak_memory_usage();
            
            lm::BPETokenizer tokenizer;
            try {
                tokenizer.train(corpus, vocab_size);
                
                auto end_time = std::chrono::high_resolution_clock::now();
                size_t end_memory = get_peak_memory_usage();
                
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                    end_time - start_time);
                size_t memory_used = (end_memory - start_memory) / (1024 * 1024);
                
                std::cout << "Training time: " << duration.count() << " ms\n";
                std::cout << "Peak memory used: " << memory_used << " MB\n";
                std::cout << "Final vocabulary size: " << tokenizer.vocab_size() << "\n";
                
                // Measure encoding performance
                std::vector<std::string> test_texts = {
                    "the quick brown fox jumps over the lazy dog",
                    "artificial intelligence and machine learning",
                    "transformer language model with attention mechanism"
                };
                
                auto encode_start = std::chrono::high_resolution_clock::now();
                size_t total_tokens = 0;
                
                for (const auto& text : test_texts) {
                    auto tokens = tokenizer.encode(text);
                    total_tokens += tokens.size();
                    
                    // Verify round-trip
                    std::string decoded = tokenizer.decode(tokens);
                    if (text != decoded) {
                        std::cout << "WARNING: Round-trip mismatch!\n";
                        std::cout << "Original: " << text << "\n";
                        std::cout << "Decoded: " << decoded << "\n";
                    }
                }
                
                auto encode_end = std::chrono::high_resolution_clock::now();
                auto encode_duration = std::chrono::duration_cast<std::chrono::microseconds>(
                    encode_end - encode_start);
                
                double encode_time_per_token = static_cast<double>(encode_duration.count()) / total_tokens;
                
                std::cout << "Encoding performance: " << encode_time_per_token << " μs/token\n";
                std::cout << "Total tokens processed: " << total_tokens << "\n";
                
            } catch (const std::exception& e) {
                std::cout << "Error during training: " << e.what() << "\n";
            }
        }
    }
    
    // Test serialization performance
    std::cout << "\n--- Serialization Performance Test ---\n";
    auto corpus = generate_test_corpus(1000, 5, 15);
    lm::BPETokenizer tokenizer;
    tokenizer.train(corpus, 1000);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    tokenizer.save("test_model.bpe");
    auto save_time = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - start_time);
    
    start_time = std::chrono::high_resolution_clock::now();
    lm::BPETokenizer loaded_tokenizer;
    loaded_tokenizer.load("test_model.bpe");
    auto load_time = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - start_time);
    
    std::cout << "Model save time: " << save_time.count() << " μs\n";
    std::cout << "Model load time: " << load_time.count() << " μs\n";
    
    // Clean up
    remove("test_model.bpe");
}

int main() {
    try {
        run_performance_test();
        std::cout << "\n=== Performance Test Completed ===\n";
    } catch (const std::exception& e) {
        std::cerr << "Performance test failed: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}
