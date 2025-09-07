#include "lm/generation/sampler.hpp"
#include "lm/tokenizer/bpe_tokenizer.hpp"
#include <iostream>
#include <fstream>
#include <chrono>

using namespace lm;

// Simple corpus for testing
std::vector<std::string> create_test_corpus() {
    return {
        "The quick brown fox jumps over the lazy dog",
        "Programming is fun with C++ and machine learning",
        "Natural language processing transforms how we interact with computers",
        "Deep learning models require large amounts of data",
        "Attention mechanisms have revolutionized neural networks"
    };
}

int main() {
    std::cout << "=== BPE Framework Generation Test ===\n\n";
    
    try {
        // Initialize tokenizer
        BPETokenizer tokenizer;
        
        // Create a small test corpus
        auto corpus = create_test_corpus();
        
        std::cout << "Training tokenizer on " << corpus.size() << " sentences...\n";
        tokenizer.train(corpus, 100); // Small vocabulary for testing
        
        std::cout << "Tokenizer vocabulary size: " << tokenizer.vocab_size() << "\n";
        std::cout << "EOS token ID: " << tokenizer.eos_token_id() << "\n";
        std::cout << "PAD token ID: " << tokenizer.pad_token_id() << "\n";
        std::cout << "UNK token ID: " << tokenizer.unk_token_id() << "\n\n";
        
        // Test encoding/decoding
        std::string test_text = "The quick brown fox";
        auto encoded = tokenizer.encode(test_text);
        auto decoded = tokenizer.decode(encoded);
        
        std::cout << "Encoding test:\n";
        std::cout << "Original: " << test_text << "\n";
        std::cout << "Encoded: ";
        for (auto token : encoded) {
            std::cout << token << " ";
        }
        std::cout << "\nDecoded: " << decoded << "\n\n";
        
        // Test different samplers
        std::cout << "\n=== Testing Samplers ===\n";
        
        // Create a simple tensor for testing samplers
        // Use explicit shape initialization to avoid Eigen assertion errors
        std::vector<size_t> shape = {10}; // 1D tensor with 10 elements
        Tensor logits(shape);
        
        // Initialize with some values - use 1D indexing
        for (int i = 0; i < 10; i++) {
            logits(i) = static_cast<float>(i) / 10.0f;
        }
        
        // Test greedy sampler
        GreedySampler greedy_sampler;
        TokenID greedy_token = greedy_sampler.sample(logits);
        std::cout << "Greedy sampler selected token: " << greedy_token << "\n";
        
        // Test random sampler
        RandomSampler random_sampler(0.8f);
        TokenID random_token = random_sampler.sample(logits);
        std::cout << "Random sampler selected token: " << random_token << "\n";
        
        // Test Top-K sampler
        TopKSampler topk_sampler(5, 0.8f);
        TokenID topk_token = topk_sampler.sample(logits);
        std::cout << "Top-K sampler selected token: " << topk_token << "\n";
        
        // Test Top-P sampler
        TopPSampler topp_sampler(0.9f, 0.8f);
        TokenID topp_token = topp_sampler.sample(logits);
        std::cout << "Top-P sampler selected token: " << topp_token << "\n\n";
        
        // Test EOS token handling
        std::cout << "=== Testing EOS Token Handling ===\n";
        std::string eos_prompt = "Test";
        auto eos_encoded = tokenizer.encode(eos_prompt);
        
        // Check if EOS token is in vocabulary
        int eos_token_id = static_cast<int>(tokenizer.eos_token_id());
        std::cout << "EOS token ID: " << eos_token_id << "\n";
        
        // Check if EOS token is in the encoded prompt
        auto eos_it = std::find(eos_encoded.begin(), eos_encoded.end(), eos_token_id);
        if (eos_it != eos_encoded.end()) {
            std::cout << "EOS token found in encoded prompt at position " 
                      << (eos_it - eos_encoded.begin()) << "\n";
        } else {
            std::cout << "EOS token not found in encoded prompt\n";
        }
        
        std::cout << "\n=== Test Completed Successfully ===\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}

