#include "lm/training/trainer.hpp"
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
        "Attention mechanisms have revolutionized neural networks",
        "The weather is beautiful today",
        "I enjoy reading books about science and technology",
        "Artificial intelligence will shape our future",
        "Machine learning algorithms can recognize patterns",
        "Neural networks are inspired by the human brain"
    };
}

// Function to measure generation performance
void benchmark_generation(LanguageModelTrainer& trainer, 
                         const std::string& prompt, 
                         Sampler& sampler, 
                         size_t max_length, 
                         size_t sequence_length,
                         int num_runs = 5) {
    std::cout << "Benchmarking generation with prompt: '" << prompt << "'\n";
    
    double total_time = 0;
    std::string generated_text;
    
    for (int i = 0; i < num_runs; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        generated_text = trainer.generate(prompt, max_length, sampler, sequence_length);
        auto end = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double> duration = end - start;
        total_time += duration.count();
        
        if (i == 0) { // Only print result once
            std::cout << "Generated: " << generated_text << "\n";
        }
    }
    
    double avg_time = total_time / num_runs;
    std::cout << "Average generation time: " << avg_time << " seconds\n";
    std::cout << "Tokens per second: " << (max_length / avg_time) << "\n\n";
}

int main() {
    std::cout << "=== BPE Framework Generation Test ===\n\n";
    
    try {
        // Initialize tokenizer
        BPETokenizer tokenizer;
        
        // Create a small test corpus
        auto corpus = create_test_corpus();
        
        std::cout << "Training tokenizer on " << corpus.size() << " sentences...\n";
        tokenizer.train(corpus, 500); // Small vocabulary for testing
        
        std::cout << "Tokenizer vocabulary size: " << tokenizer.vocab_size() << "\n";
        std::cout << "EOS token ID: " << tokenizer.eos_token_id() << "\n";
        std::cout << "PAD token ID: " << tokenizer.pad_token_id() << "\n";
        std::cout << "UNK token ID: " << tokenizer.unk_token_id() << "\n\n";
        
        // Initialize model trainer
        std::cout << "Initializing language model...\n";
        LanguageModelTrainer trainer(tokenizer, 64, 128, 2); // Small model for testing
        
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
        
        // Train the model (briefly)
        std::cout << "Training model (5 epochs)...\n";
        trainer.train(corpus, 5, 2, 16, 0.2f, 1, 2);
        
        // Test different samplers
        std::string prompt = "The future of";
        
        GreedySampler greedy_sampler;
        RandomSampler random_sampler(0.8f);
        TopKSampler topk_sampler(10, 0.8f);
        TopPSampler topp_sampler(0.9f, 0.8f);
        
        std::cout << "\n=== Testing Different Samplers ===\n";
        
        std::cout << "\nGreedy sampling:\n";
        benchmark_generation(trainer, prompt, greedy_sampler, 20, 16);
        
        std::cout << "Random sampling (temperature=0.8):\n";
        benchmark_generation(trainer, prompt, random_sampler, 20, 16);
        
        std::cout << "Top-K sampling (k=10, temperature=0.8):\n";
        benchmark_generation(trainer, prompt, topk_sampler, 20, 16);
        
        std::cout << "Top-P sampling (p=0.9, temperature=0.8):\n";
        benchmark_generation(trainer, prompt, topp_sampler, 20, 16);
        
        // Test batch generation
        std::cout << "=== Testing Batch Generation ===\n";
        std::vector<std::string> prompts = {
            "The future of",
            "Machine learning",
            "Artificial intelligence"
        };
        
        auto batch_results = trainer.generate_batch(prompts, 15, greedy_sampler, 16, 2);
        
        for (size_t i = 0; i < prompts.size(); i++) {
            std::cout << "Prompt: '" << prompts[i] << "'\n";
            std::cout << "Generated: '" << batch_results[i] << "'\n\n";
        }
        
        // Test EOS token handling
        std::cout << "=== Testing EOS Token Handling ===\n";
        std::string eos_prompt = "Test";
        std::vector<int> tokens = trainer.generate_and_return_tokens(eos_prompt, 50, greedy_sampler, 16);
        
        std::cout << "Generated " << tokens.size() << " tokens\n";
        
        // Check if EOS token was generated
        int eos_token_id = static_cast<int>(tokenizer.eos_token_id());
        auto eos_it = std::find(tokens.begin(), tokens.end(), eos_token_id);
        
        if (eos_it != tokens.end()) {
            std::cout << "EOS token found at position " << (eos_it - tokens.begin()) << "\n";
            std::cout << "Text before EOS: " << tokenizer.decode(
                std::vector<TokenID>(tokens.begin(), eos_it)) << "\n";
        } else {
            std::cout << "EOS token not generated\n";
        }
        
        std::cout << "\n=== Test Completed Successfully ===\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}
