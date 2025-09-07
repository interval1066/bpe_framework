#include "lm/generation/sampler.hpp"
#include "lm/tokenizer/bpe_tokenizer.hpp"
#include <iostream>
#include <cassert>

using namespace lm;

void test_samplers() {
    std::cout << "=== Testing Samplers ===" << std::endl;
    
    // Create a simple logits tensor
    std::vector<size_t> shape = {10}; // Vocabulary size 10
    Tensor logits(shape);
    
    // Set up logits (highest probability at index 3)
    for (size_t i = 0; i < 10; i++) {
        logits(i) = (i == 3) ? 5.0f : 1.0f; // Index 3 has highest probability
    }
    
    // Test GreedySampler
    GreedySampler greedy_sampler;
    int greedy_token = greedy_sampler.sample(logits);
    std::cout << "Greedy sampler selected token: " << greedy_token << std::endl;
    assert(greedy_token == 3); // Should always select the highest probability
    
    // Test RandomSampler
    RandomSampler random_sampler(1.0f); // Temperature 1.0
    int random_token = random_sampler.sample(logits);
    std::cout << "Random sampler selected token: " << random_token << std::endl;
    assert(random_token >= 0 && random_token < 10); // Should be a valid token
    
    // Test TopKSampler
    TopKSampler topk_sampler(3, 1.0f); // Top 3, temperature 1.0
    int topk_token = topk_sampler.sample(logits);
    std::cout << "Top-K sampler selected token: " << topk_token << std::endl;
    assert(topk_token >= 0 && topk_token < 10); // Should be a valid token
    
    // Test TopPSampler
    TopPSampler topp_sampler(0.9f, 1.0f); // Top-P 0.9, temperature 1.0
    int topp_token = topp_sampler.sample(logits);
    std::cout << "Top-P sampler selected token: " << topp_token << std::endl;
    assert(topp_token >= 0 && topp_token < 10); // Should be a valid token
    
    std::cout << "All samplers passed basic tests!" << std::endl;
}

void test_tokenizer_generation() {
    std::cout << "\n=== Testing Tokenizer Generation ===" << std::endl;
    
    // Create a simple tokenizer
    BPETokenizer tokenizer;
    // Train on a small corpus
    std::vector<std::string> corpus = {
        "hello world",
        "test sentence",
        "another example"
    };

    tokenizer.train(corpus, 50); // Small vocabulary

    // Test encoding/decoding
    std::string test_text = "hello test";
    std::vector<TokenID> encoded = tokenizer.encode(test_text);
    std::string decoded = tokenizer.decode(encoded);

    std::cout << "Original: " << test_text << std::endl;
    std::cout << "Encoded: ";
    for (auto token : encoded) {
        std::cout << token << " ";
    }
    std::cout << std::endl;
    std::cout << "Decoded: " << decoded << std::endl;
    
    // Basic sanity check
    assert(encoded.size() > 0);
    assert(!decoded.empty());
    
    std::cout << "Tokenizer generation test passed!" << std::endl;
}

void test_temperature_effects() {
    std::cout << "\n=== Testing Temperature Effects ===" << std::endl;
    
    // Create a simple logits tensor
    std::vector<size_t> shape = {5}; // Vocabulary size 5
    Tensor logits(shape);
    
    // Set up logits
    for (size_t i = 0; i < 5; i++) {
        logits(i) = static_cast<float>(i);
    }
    
    // Test different temperature values
    RandomSampler high_temp_sampler(2.0f); // High temperature
    RandomSampler low_temp_sampler(0.5f);  // Low temperature
    
    int high_temp_token = high_temp_sampler.sample(logits);
    int low_temp_token = low_temp_sampler.sample(logits);
    
    std::cout << "High temperature (2.0) selected token: " << high_temp_token << std::endl;
    std::cout << "Low temperature (0.5) selected token: " << low_temp_token << std::endl;
    
    // Both should be valid tokens
    assert(high_temp_token >= 0 && high_temp_token < 5);
    assert(low_temp_token >= 0 && low_temp_token < 5);
    
    std::cout << "Temperature effects test passed!" << std::endl;
}

void test_sampler_consistency() {
    std::cout << "\n=== Testing Sampler Consistency ===" << std::endl;
    
    // Create a simple logits tensor
    std::vector<size_t> shape = {5}; // Vocabulary size 5
    Tensor logits(shape);
    
    // Set up logits with one clear winner
    logits(0) = 1.0f;
    logits(1) = 1.0f;
    logits(2) = 10.0f; // Clear winner
    logits(3) = 1.0f;
    logits(4) = 1.0f;
    
    // Greedy sampler should always pick the same token
    GreedySampler greedy_sampler;
    int first_token = greedy_sampler.sample(logits);
    
    // Test multiple times
    for (int i = 0; i < 10; i++) {
        int token = greedy_sampler.sample(logits);
        assert(token == first_token);
    }
    
    std::cout << "Greedy sampler is consistent (always selects token " << first_token << ")" << std::endl;
    std::cout << "Sampler consistency test passed!" << std::endl;
}

int main() {
    std::cout << "Starting sampler functionality tests..." << std::endl;
    
    try {
        test_samplers();
        test_tokenizer_generation();
        test_temperature_effects();
        test_sampler_consistency();
        
        std::cout << "\n=== All Tests Passed! ===" << std::endl;
        std::cout << "Sampler functionality is working correctly." << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with error: " << e.what() << std::endl;
        return 1;
    }
}

