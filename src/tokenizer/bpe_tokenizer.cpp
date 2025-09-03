#include "lm/tokenizer/bpe_tokenizer.hpp"
#include "lm/tokenizer/unicode_utils.hpp"
#include <fstream>
#include <sstream>
#include <queue>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <sys/resource.h>
#include <vector>
#include <memory>
#include <unordered_map>

// Add CPU-specific optimizations
#ifdef __SSE4_2__
#include <nmmintrin.h>  // For SSE4.2 intrinsics
#endif

namespace lm {

// Custom hash function for pair<TokenID, TokenID>
struct PairHash {
    size_t operator()(const std::pair<TokenID, TokenID>& p) const {
        return (static_cast<size_t>(p.first) << 16) | p.second;
    }
};

// Memory tracking function
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

// String interning class
class StringInternPool {
    std::unordered_map<std::string, std::shared_ptr<const std::string>> pool;
    
public:
    std::shared_ptr<const std::string> intern(const std::string& str) {
        auto it = pool.find(str);
        if (it != pool.end()) {
            return it->second;
        }
        
        auto shared_str = std::make_shared<std::string>(str);
        pool[str] = shared_str;
        return shared_str;
    }
    
    void clear() {
        pool.clear();
    }
};

// Unicode processing cache
class UnicodeCache {
private:
    mutable std::unordered_map<std::string, std::string> normalization_cache;
    mutable std::unordered_map<std::string, std::vector<std::string>> split_cache;
    
public:
    const std::string& get_normalized(const std::string& text) const {
        auto it = normalization_cache.find(text);
        if (it != normalization_cache.end()) {
            return it->second;
        }
        
        auto normalized = unicode::normalize(text);
        auto result = normalization_cache.emplace(text, std::move(normalized));
        return result.first->second;
    }
    
    const std::vector<std::string>& get_split(const std::string& text) const {
        auto it = split_cache.find(text);
        if (it != split_cache.end()) {
            return it->second;
        }
        
        auto split = unicode::unicode_split(text);
        auto result = split_cache.emplace(text, std::move(split));
        return result.first->second;
    }
    
    void clear() const {
        normalization_cache.clear();
        split_cache.clear();
    }
};

// UTF-8 validation - using C++ implementation only
static bool is_valid_utf8_asm(const char* str, size_t length) {
    // Simple UTF-8 validation
    for (size_t i = 0; i < length; i++) {
        unsigned char c = str[i];
        if (c > 0x7F) {  // Non-ASCII character
            // Check if it's a valid UTF-8 start byte
            if (c < 0xC2 || c > 0xF4) return false;
            
            // Check continuation bytes
            int following_bytes = 0;
            if ((c & 0xE0) == 0xC0) following_bytes = 1;
            else if ((c & 0xF0) == 0xE0) following_bytes = 2;
            else if ((c & 0xF8) == 0xF0) following_bytes = 3;
            
            // Check if we have enough bytes
            if (i + following_bytes >= length) return false;
            
            // Check continuation bytes
            for (int j = 1; j <= following_bytes; j++) {
                if ((str[i + j] & 0xC0) != 0x80) return false;
            }
            
            i += following_bytes;
        }
    }
    return true;
}

struct BPETokenizer::Impl {
    std::unordered_map<std::string, TokenID> vocab;
    std::unordered_map<TokenID, std::string> inv_vocab;
    std::unordered_map<std::pair<TokenID, TokenID>, TokenID, PairHash> merges;
    std::unordered_map<std::string, TokenID> special_tokens;
    std::string unknown_token = "<unk>";
    TokenID unknown_token_id = 0;
    TokenID next_token_id = 0;
    bool normalization_enabled = true;
    bool byte_fallback_enabled = true;
    StringInternPool string_pool;
    mutable UnicodeCache unicode_cache;  // Made mutable
    bool cache_enabled = true;
    
    // Special token IDs
    TokenID eos_token_id = 0;
    TokenID pad_token_id = 0;
    TokenID unk_token_id = 0;
    
    // Helper functions
    std::vector<std::string> split_text(const std::string& text) const;
    std::vector<TokenID> word_to_token_ids(const std::string& word) const;
    void initialize_vocab();
    void count_word_frequencies(const std::vector<std::string>& words,
                               std::unordered_map<std::string, int>& word_counts) const;
    void get_pair_counts(const std::unordered_map<std::string, int>& word_counts,
                        std::unordered_map<std::pair<TokenID, TokenID>, int, PairHash>& pair_counts) const;
    void perform_merge(const std::pair<TokenID, TokenID>& pair, TokenID new_token_id,
                      std::unordered_map<std::string, int>& word_counts);
    
    // Handle invalid UTF-8
    std::vector<TokenID> handle_invalid_utf8(const std::string& text) const;
    
    // CPU Optimization: Batch processing
    void process_string_batch(const std::vector<std::string>& batch);
    
    // Cache management
    void enable_caching(bool enable) {
        cache_enabled = enable;
        if (!enable) {
            unicode_cache.clear();
        }
    }
};

BPETokenizer::BPETokenizer() : pimpl_(new Impl) {
    pimpl_->initialize_vocab();
}

BPETokenizer::~BPETokenizer() = default;

void BPETokenizer::Impl::initialize_vocab() {
    // Preallocate with more realistic sizes
    vocab.reserve(65536);
    inv_vocab.reserve(65536);
    special_tokens.reserve(256);
    merges.reserve(30000);
    
    // Add bytes with optimized insertion
    for (int i = 0; i < 256; i++) {
        std::string token(1, static_cast<char>(i));
        vocab.emplace(token, next_token_id);
        inv_vocab.emplace(next_token_id++, std::move(token));
    }
    
    // Add special tokens
    vocab["<unk>"] = next_token_id;
    inv_vocab[next_token_id] = "<unk>";
    special_tokens["<unk>"] = next_token_id;
    unk_token_id = next_token_id++;
    
    vocab["<pad>"] = next_token_id;
    inv_vocab[next_token_id] = "<pad>";
    special_tokens["<pad>"] = next_token_id;
    pad_token_id = next_token_id++;
    
    vocab["<eos>"] = next_token_id;
    inv_vocab[next_token_id] = "<eos>";
    special_tokens["<eos>"] = next_token_id;
    eos_token_id = next_token_id++;
    
    // Set unknown token ID
    unknown_token_id = unk_token_id;
}

std::vector<std::string> BPETokenizer::Impl::split_text(const std::string& text) const {
    if (normalization_enabled) {
        if (cache_enabled) {
            return unicode_cache.get_split(unicode_cache.get_normalized(text));
        } else {
            std::string normalized = unicode::normalize(text);
            return unicode::unicode_split(normalized);
        }
    } else {
        std::vector<std::string> words;
        std::istringstream iss(text);  // Fixed the typo here
        std::string word;
        
        // Preallocate based on text size
        words.reserve(text.size() / 6); // Average word length ~6 characters
        
        while (iss >> word) {
            words.push_back(std::move(word));
        }
        
        return words;
    }
}

std::vector<TokenID> BPETokenizer::Impl::word_to_token_ids(const std::string& word) const {
    // Hot path optimization: use local references to avoid repeated pointer dereferencing
    const auto& local_vocab = vocab;
    const TokenID local_unknown_id = unknown_token_id;
    const bool local_byte_fallback = byte_fallback_enabled;
    const bool local_normalization = normalization_enabled;
    
    std::vector<TokenID> tokens;
    
    // Preallocate based on word size (assuming most characters will be tokens)
    tokens.reserve(word.size());
    
    if (local_normalization) {
        std::string normalized;
        if (cache_enabled) {
            normalized = unicode_cache.get_normalized(word);
        } else {
            normalized = unicode::normalize(word);
        }
        
        std::vector<std::string> characters;
        if (cache_enabled) {
            characters = unicode_cache.get_split(normalized);
        } else {
            characters = unicode::split_on_character_boundaries(normalized);
        }
        
        for (const auto& character : characters) {
            auto it = local_vocab.find(character);
            if (it != local_vocab.end()) {
                tokens.push_back(it->second);
            } else if (local_byte_fallback) {
                // If character not found, try to split into bytes
                for (unsigned char c : character) {
                    std::string byte_str(1, static_cast<char>(c));
                    auto byte_it = local_vocab.find(byte_str);
                    if (byte_it != local_vocab.end()) {
                        tokens.push_back(byte_it->second);
                    } else {
                        tokens.push_back(local_unknown_id);
                    }
                }
            } else {
                tokens.push_back(local_unknown_id);
            }
        }
    } else {
        // Non-Unicode mode: treat as ASCII
        for (char c : word) {
            std::string token(1, c);
            auto it = local_vocab.find(token);
            if (it != local_vocab.end()) {
                tokens.push_back(it->second);
            } else {
                tokens.push_back(local_unknown_id);
            }
        }
    }
    
    return tokens;
}

void BPETokenizer::Impl::count_word_frequencies(
    const std::vector<std::string>& words,
    std::unordered_map<std::string, int>& word_counts) const {
    
    // Preallocate based on expected unique words
    word_counts.reserve(words.size() / 10); // Assume 10% unique words
    
    for (const auto& word : words) {
        // Use emplace for more efficient insertion
        auto result = word_counts.emplace(word, 1);
        if (!result.second) {
            result.first->second++;
        }
    }
}

void BPETokenizer::Impl::perform_merge(const std::pair<TokenID, TokenID>& pair, TokenID new_token_id,
                                      std::unordered_map<std::string, int>& word_counts) {
    std::string new_token = inv_vocab.at(pair.first) + inv_vocab.at(pair.second);
    auto shared_token = string_pool.intern(new_token);
    
    // Add new token to vocabulary
    vocab[*shared_token] = new_token_id;
    inv_vocab[new_token_id] = *shared_token;
    
    // Record the merge
    merges[pair] = new_token_id;
    
    // Update word counts with new merges - more efficient implementation
    std::unordered_map<std::string, int> new_word_counts;
    new_word_counts.reserve(word_counts.size());
    
    const std::string& first_token = inv_vocab.at(pair.first);
    const std::string& second_token = inv_vocab.at(pair.second);
    const std::string& merged_token = *shared_token;
    
    for (const auto& [word, count] : word_counts) {
        std::string new_word;
        new_word.reserve(word.size()); // Pre-allocate
        
        size_t pos = 0;
        while (pos < word.size()) {
            // Check if we found the pair to merge
            if (pos <= word.size() - first_token.size() - second_token.size() &&
                word.compare(pos, first_token.size(), first_token) == 0 &&
                word.compare(pos + first_token.size(), second_token.size(), second_token) == 0) {
                new_word += merged_token;
                pos += first_token.size() + second_token.size();
            } else {
                new_word += word[pos++];
            }
        }
        
        if (new_word != word) {
            new_word_counts[new_word] += count;
        } else {
            new_word_counts[word] += count;
        }
    }
    
    word_counts = std::move(new_word_counts);
}

std::vector<TokenID> BPETokenizer::Impl::handle_invalid_utf8(const std::string& text) const {
    std::vector<TokenID> tokens;
    tokens.reserve(text.size());
    
    for (size_t i = 0; i < text.size(); i++) {
        unsigned char c = text[i];
        
        // If it's a valid ASCII character, encode normally
        if (c <= 0x7F) {
            std::string char_str(1, static_cast<char>(c));
            if (auto it = vocab.find(char_str); it != vocab.end()) {
                tokens.push_back(it->second);
            } else {
                tokens.push_back(unknown_token_id);
            }
        } else {
            // Invalid byte, use byte fallback or unknown token
            if (byte_fallback_enabled) {
                // Encode each byte individually
                std::string byte_str(1, static_cast<char>(c));
                if (auto it = vocab.find(byte_str); it != vocab.end()) {
                    tokens.push_back(it->second);
                } else {
                    tokens.push_back(unknown_token_id);
                }
            } else {
                tokens.push_back(unknown_token_id);
            }
        }
    }
    
    return tokens;
}

void BPETokenizer::train(const std::vector<std::string>& corpus, size_t vocab_size) {
    size_t start_memory = get_peak_memory_usage();
    
    if (corpus.empty()) {
        throw std::invalid_argument("Corpus cannot be empty");
    }
    
    // Disable caching during training as vocabulary changes frequently
    pimpl_->enable_caching(false);
    
    // Validate all input texts before training
    for (const auto& text : corpus) {
        if (!BPETokenizer::is_valid_utf8_asm(text.data(), text.size())) {
            std::cerr << "Warning: Invalid UTF-8 in training corpus: " << text << std::endl;
            // Skip invalid text
            continue;
        }
    }
    
    // Split text into words
    std::vector<std::string> words;
    words.reserve(corpus.size() * 10); // Estimate average words per sentence
    
    for (const auto& text : corpus) {
        auto text_words = pimpl_->split_text(text);
        words.insert(words.end(), text_words.begin(), text_words.end());
    }
    
    // Count word frequencies using strings
    std::unordered_map<std::string, int> word_counts;
    pimpl_->count_word_frequencies(words, word_counts);
    
    // BPE training algorithm with safety limit
    int iteration = 0;
    int max_iterations = 10000;
    
    // Pre-allocate pair counts
    std::unordered_map<std::pair<TokenID, TokenID>, int, PairHash> pair_counts;
    pair_counts.reserve(1000000); // Reserve space for 1M pairs
    
    while (pimpl_->vocab.size() < vocab_size && iteration < max_iterations) {
        // Count pairs
        pair_counts.clear();
        pimpl_->get_pair_counts(word_counts, pair_counts);
        
        if (pair_counts.empty()) {
            std::cout << "No more pairs to merge. Stopping early." << std::endl;
            break;
        }
        
        // Find most frequent pair
        auto max_pair = std::max_element(
            pair_counts.begin(), pair_counts.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; }
        );
        
        // Debug output
        if (iteration % 100 == 0) {
            std::cout << "Iteration " << iteration 
                      << ", Vocab size: " << pimpl_->vocab.size()
                      << ", Most frequent pair: (" << max_pair->first.first 
                      << "," << max_pair->first.second << ") count: " 
                      << max_pair->second << std::endl;
        }
        
        // Perform merge
        pimpl_->perform_merge(max_pair->first, pimpl_->next_token_id, word_counts);
        pimpl_->next_token_id++;
        iteration++;
        
        // Periodically check memory usage and clean up
        if (iteration % 500 == 0) {
            size_t current_memory = get_peak_memory_usage();
            std::cout << "Memory after " << iteration << " iterations: " 
                      << (current_memory - start_memory) / (1024 * 1024) << "MB\n";
            
            // Clean up string pool to save memory
            pimpl_->string_pool.clear();
        }
    }
    
    if (iteration >= max_iterations) {
        std::cout << "Reached maximum iterations. Stopping training." << std::endl;
    }
    
    // Re-enable caching after training
    pimpl_->enable_caching(true);
    
    size_t end_memory = get_peak_memory_usage();
    std::cout << "Training completed in " << iteration << " iterations\n";
    std::cout << "Peak memory used: " << (end_memory - start_memory) / (1024 * 1024) << "MB\n";
    std::cout << "Final vocabulary size: " << pimpl_->vocab.size() << std::endl;
}

void BPETokenizer::Impl::get_pair_counts(
    const std::unordered_map<std::string, int>& word_counts,
    std::unordered_map<std::pair<TokenID, TokenID>, int, PairHash>& pair_counts) const {
    
    // Pre-allocate memory for better performance
    pair_counts.reserve(word_counts.size() * 10);
    
    for (const auto& [word, count] : word_counts) {
        auto tokens = word_to_token_ids(word);
        for (size_t i = 0; i < tokens.size() - 1; i++) {
            auto pair = std::make_pair(tokens[i], tokens[i+1]);
            pair_counts[pair] += count;
        }
    }
}

// Add these missing member function implementations at the end of the bpe_tokenizer.cpp file

size_t BPETokenizer::vocab_size() const {
    return pimpl_->vocab.size();
}

std::vector<TokenID> BPETokenizer::encode(const std::string& text) const {
    // Validate UTF-8 before processing
    if (!BPETokenizer::is_valid_utf8_asm(text.data(), text.size())) {
        // Handle invalid UTF-8
        if (pimpl_->byte_fallback_enabled) {
            return pimpl_->handle_invalid_utf8(text);
        } else {
            return {pimpl_->unknown_token_id};
        }
    }
    
    auto words = pimpl_->split_text(text);
    std::vector<TokenID> tokens;
    tokens.reserve(text.size() * 2); // Pre-allocate based on text size
    
    for (const auto& word : words) {
        auto word_tokens = pimpl_->word_to_token_ids(word);
        
        // Apply BPE merges more efficiently
        bool changed;
        do {
            changed = false;
            for (size_t i = 0; i < word_tokens.size() - 1; i++) {
                auto pair = std::make_pair(word_tokens[i], word_tokens[i+1]);
                if (auto it = pimpl_->merges.find(pair); it != pimpl_->merges.end()) {
                    word_tokens[i] = it->second;
                    word_tokens.erase(word_tokens.begin() + i + 1);
                    changed = true;
                    break;
                }
            }
        } while (changed);
        
        tokens.insert(tokens.end(), word_tokens.begin(), word_tokens.end());
    }
    
    return tokens;
}

std::string BPETokenizer::decode(const std::vector<TokenID>& tokens) const {
    std::string text;
    text.reserve(tokens.size() * 3); // Estimate average token length
    
    for (TokenID token_id : tokens) {
        if (pimpl_->inv_vocab.find(token_id) != pimpl_->inv_vocab.end()) {
            text += pimpl_->inv_vocab.at(token_id);
        } else {
            text += pimpl_->unknown_token;
        }
    }
    return text;
}

bool BPETokenizer::save(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return false;
    }
    
    // Save vocabulary
    file << pimpl_->vocab.size() << "\n";
    for (const auto& [token, id] : pimpl_->vocab) {
        file << id << " " << token << "\n";
    }
    
    // Save merges
    file << pimpl_->merges.size() << "\n";
    for (const auto& [pair, new_id] : pimpl_->merges) {
        file << pair.first << " " << pair.second << " " << new_id << "\n";
    }
    
    return true;
}

bool BPETokenizer::load(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        return false;
    }
    
    // Clear existing data
    pimpl_->vocab.clear();
    pimpl_->inv_vocab.clear();
    pimpl_->merges.clear();
    
    // Load vocabulary
    size_t vocab_size;
    file >> vocab_size;
    for (size_t i = 0; i < vocab_size; i++) {
        TokenID id;
        std::string token;
        file >> id;
        std::getline(file, token);
        // Remove leading space
        if (!token.empty() && token[0] == ' ') {
            token = token.substr(1);
        }
        pimpl_->vocab[token] = id;
        pimpl_->inv_vocab[id] = token;
    }
    
    // Load merges
    size_t merge_count;
    file >> merge_count;
    for (size_t i = 0; i < merge_count; i++) {
        TokenID first, second, new_id;
        file >> first >> second >> new_id;
        pimpl_->merges[{first, second}] = new_id;
    }
    
    return true;
}

// Special token method implementations
TokenID BPETokenizer::eos_token_id() const { 
    return pimpl_->eos_token_id; 
}

void BPETokenizer::set_eos_token_id(TokenID id) { 
    pimpl_->eos_token_id = id; 
}

TokenID BPETokenizer::pad_token_id() const { 
    return pimpl_->pad_token_id; 
}

void BPETokenizer::set_pad_token_id(TokenID id) { 
    pimpl_->pad_token_id = id; 
}

TokenID BPETokenizer::unk_token_id() const { 
    return pimpl_->unk_token_id; 
}

void BPETokenizer::set_unk_token_id(TokenID id) { 
    pimpl_->unk_token_id = id; 
}

void BPETokenizer::add_special_token(const std::string& token, TokenID id) {
    pimpl_->vocab[token] = id;
    pimpl_->inv_vocab[id] = token;
    pimpl_->special_tokens[token] = id;
    
    // Update the specific token ID if it matches known types
    if (token == "<eos>" || token == "</s>") {
        pimpl_->eos_token_id = id;
    } else if (token == "<pad>") {
        pimpl_->pad_token_id = id;
    } else if (token == "<unk>") {
        pimpl_->unk_token_id = id;
    }
}

// UTF-8 validation method implementation
bool BPETokenizer::is_valid_utf8_asm(const char* str, size_t length) {
    return is_valid_utf8_asm(str, length);
}

} // namespace lm
