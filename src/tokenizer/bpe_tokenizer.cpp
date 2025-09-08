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
#include <iomanip>

// Add CPU-specific optimizations
#ifdef __SSE4_2__
#include <nmmintrin.h>  // For SSE4.2 intrinsics
#endif

namespace lm {

struct VectorHash {
    size_t operator()(const std::vector<TokenID>& vec) const {
        size_t seed = vec.size();
        for (const auto& token : vec) {
            seed ^= token + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

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
namespace {
bool is_valid_utf8_impl(const char* str, size_t length) {
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
} // namespace

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
    bool debug_logging = false;  // Added debug logging flag
    
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
    void get_pair_counts_from_sequences(const std::vector<std::pair<std::vector<TokenID>, int>>& tokenized_corpus,
                                       std::unordered_map<std::pair<TokenID, TokenID>, int, PairHash>& pair_counts) const;
    void perform_merge_on_sequences(const std::pair<TokenID, TokenID>& pair, TokenID new_token_id,
                                   std::vector<std::pair<std::vector<TokenID>, int>>& tokenized_corpus);

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
    
    // Debug logging methods
    void log_encode_start(const std::string& text) const;
    void log_word_split(const std::vector<std::string>& words) const;
    void log_word_tokens(const std::string& word, const std::vector<TokenID>& tokens) const;
    void log_merge_attempt(size_t pos, TokenID first, TokenID second, bool found) const;
    void log_merge_result(const std::vector<TokenID>& tokens) const;
    void log_final_tokens(const std::vector<TokenID>& tokens) const;
    void log_decode_start(const std::vector<TokenID>& tokens) const;
    void log_token_decoding(TokenID token_id, const std::string& decoded) const;
    void log_final_decoding(const std::string& text) const;
};

// Debug logging implementations
void BPETokenizer::Impl::log_encode_start(const std::string& text) const {
    if (!debug_logging) return;
    std::cout << "[ENCODE] Starting encoding of text: '" << text << "'" << std::endl;
}

void BPETokenizer::Impl::get_pair_counts_from_sequences(
    const std::vector<std::pair<std::vector<TokenID>, int>>& tokenized_corpus,
    std::unordered_map<std::pair<TokenID, TokenID>, int, PairHash>& pair_counts) const {
    
    pair_counts.clear();
    
    for (const auto& [sequence, count] : tokenized_corpus) {
        for (size_t i = 0; i < sequence.size() - 1; i++) {
            auto pair = std::make_pair(sequence[i], sequence[i+1]);
            pair_counts[pair] += count;
        }
    }
}

void BPETokenizer::Impl::log_word_split(const std::vector<std::string>& words) const {
    if (!debug_logging) return;
    std::cout << "[ENCODE] Split into " << words.size() << " words: ";
    for (size_t i = 0; i < words.size(); i++) {
        std::cout << "[" << i << "]='" << words[i] << "' ";
    }
    std::cout << std::endl;
}

void BPETokenizer::Impl::log_word_tokens(const std::string& word, const std::vector<TokenID>& tokens) const {
    if (!debug_logging) return;
    std::cout << "[ENCODE] Word '" << word << "' → Tokens: ";
    for (TokenID id : tokens) {
        std::cout << id << " ('" << (inv_vocab.count(id) ? inv_vocab.at(id) : "<?>") << "') ";
    }
    std::cout << std::endl;
}

void BPETokenizer::Impl::log_merge_attempt(size_t pos, TokenID first, TokenID second, bool found) const {
    if (!debug_logging) return;
    std::string first_str = inv_vocab.count(first) ? inv_vocab.at(first) : "<?>";
    std::string second_str = inv_vocab.count(second) ? inv_vocab.at(second) : "<?>";
    std::cout << "[ENCODE] Checking pair at position " << pos << ": (" 
              << first << ":'" << first_str << "', " 
              << second << ":'" << second_str << "') - " 
              << (found ? "FOUND" : "NOT FOUND") << std::endl;
}

void BPETokenizer::Impl::log_merge_result(const std::vector<TokenID>& tokens) const {
    if (!debug_logging) return;
    std::cout << "[ENCODE] After merge: ";
    for (TokenID id : tokens) {
        std::cout << id << " ('" << (inv_vocab.count(id) ? inv_vocab.at(id) : "<?>") << "') ";
    }
    std::cout << std::endl;
}

void BPETokenizer::Impl::log_final_tokens(const std::vector<TokenID>& tokens) const {
    if (!debug_logging) return;
    std::cout << "[ENCODE] Final tokens: ";
    for (TokenID id : tokens) {
        std::cout << id << " ";
    }
    std::cout << std::endl;
    std::cout << "[ENCODE] Final tokens with text: ";
    for (TokenID id : tokens) {
        std::cout << id << ":'" << (inv_vocab.count(id) ? inv_vocab.at(id) : "<?>") << "' ";
    }
    std::cout << std::endl;
}

void BPETokenizer::Impl::log_decode_start(const std::vector<TokenID>& tokens) const {
    if (!debug_logging) return;
    std::cout << "[DECODE] Starting decoding of " << tokens.size() << " tokens: ";
    for (TokenID id : tokens) {
        std::cout << id << " ";
    }
    std::cout << std::endl;
}

void BPETokenizer::Impl::log_token_decoding(TokenID token_id, const std::string& decoded) const {
    if (!debug_logging) return;
    std::string token_text = inv_vocab.count(token_id) ? inv_vocab.at(token_id) : "<?>";
    std::cout << "[DECODE] Token " << token_id << ":'" << token_text << "' → '" << decoded << "'" << std::endl;
}

void BPETokenizer::Impl::log_final_decoding(const std::string& text) const {
    if (!debug_logging) return;
    std::cout << "[DECODE] Final result: '" << text << "'" << std::endl;
}

// Add debug methods to the BPETokenizer class
void BPETokenizer::enable_debug_logging(bool enable) {
    pimpl_->debug_logging = enable;
}

void BPETokenizer::dump_vocabulary() const {
    std::cout << "=== VOCABULARY DUMP ===" << std::endl;
    std::cout << "Size: " << pimpl_->vocab.size() << std::endl;
    
    // Create a sorted list for better readability
    std::vector<std::pair<std::string, TokenID>> sorted_vocab;
    for (const auto& entry : pimpl_->vocab) {
        sorted_vocab.emplace_back(entry.first, entry.second);
    }
    
    std::sort(sorted_vocab.begin(), sorted_vocab.end(),
        [](const auto& a, const auto& b) { return a.second < b.second; });
    
    for (const auto& entry : sorted_vocab) {
        std::string display = entry.first;
        // Replace non-printable characters
        for (char& c : display) {
            if (c < 32 || c > 126) {
                c = '?';
            }
        }
        std::cout << std::setw(6) << entry.second << ": '" << display << "'";
        if (entry.first != display) {
            std::cout << " (original: ";
            for (unsigned char c : entry.first) {
                if (c >= 32 && c <= 126) {
                    std::cout << c;
                } else {
                    std::cout << "\\x" << std::hex << std::setw(2) << std::setfill('0') 
                              << static_cast<int>(c) << std::dec;
                }
            }
            std::cout << ")";
        }
        std::cout << std::endl;
    }
    std::cout << "=== END VOCABULARY DUMP ===" << std::endl;
}

void BPETokenizer::dump_merges() const {
    std::cout << "=== MERGES DUMP ===" << std::endl;
    std::cout << "Number of merges: " << pimpl_->merges.size() << std::endl;
    
    for (const auto& merge : pimpl_->merges) {
        const auto& pair = merge.first;
        TokenID new_id = merge.second;
        
        std::string first_str = pimpl_->inv_vocab.count(pair.first) 
            ? pimpl_->inv_vocab.at(pair.first) : "<?>";
        std::string second_str = pimpl_->inv_vocab.count(pair.second) 
            ? pimpl_->inv_vocab.at(pair.second) : "<?>";
        std::string new_str = pimpl_->inv_vocab.count(new_id) 
            ? pimpl_->inv_vocab.at(new_id) : "<?>";
            
        std::cout << "(" << pair.first << ":'" << first_str << "', " 
                  << pair.second << ":'" << second_str << "') → " 
                  << new_id << ":'" << new_str << "'" << std::endl;
    }
    std::cout << "=== END MERGES DUMP ===" << std::endl;
}

BPETokenizer::BPETokenizer() : pimpl_(new Impl) {
    pimpl_->initialize_vocab();
}

BPETokenizer::~BPETokenizer() = default;

void BPETokenizer::Impl::initialize_vocab() {
    vocab.reserve(65536);
    inv_vocab.reserve(65536);
    special_tokens.reserve(256);
    merges.reserve(30000);
    
    // Add bytes
    for (int i = 0; i < 256; i++) {
        std::string token(1, static_cast<char>(i));
        vocab.emplace(token, next_token_id);
        inv_vocab.emplace(next_token_id++, std::move(token));
    }
    
    // Add space token
    vocab[" "] = next_token_id;
    inv_vocab[next_token_id] = " ";
    next_token_id++;
    
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
    
    unknown_token_id = unk_token_id;
}

void BPETokenizer::Impl::perform_merge_on_sequences(
    const std::pair<TokenID, TokenID>& pair, 
    TokenID new_token_id,
    std::vector<std::pair<std::vector<TokenID>, int>>& tokenized_corpus) {
    
    // Create new token
    std::string new_token = this->inv_vocab.at(pair.first) + this->inv_vocab.at(pair.second);
    
    // Add to vocabulary
    this->vocab[new_token] = new_token_id;
    this->inv_vocab[new_token_id] = new_token;
    this->merges[pair] = new_token_id;
    
    // Apply merge to all sequences
    for (auto& [sequence, count] : tokenized_corpus) {
        std::vector<TokenID> new_sequence;
        new_sequence.reserve(sequence.size());
        
        for (size_t i = 0; i < sequence.size(); i++) {
            if (i < sequence.size() - 1 && 
                sequence[i] == pair.first && 
                sequence[i+1] == pair.second) {
                new_sequence.push_back(new_token_id);
                i++; // Skip the next token
            } else {
                new_sequence.push_back(sequence[i]);
            }
        }
        
        sequence = std::move(new_sequence);
    }
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
        std::istringstream iss(text);
        std::string word;
        
        // Preallocate based on text size
        words.reserve(text.size() / 6); // Average word length ~6 characters
        
        while (iss >> word) {
            words.push_back(std::move(word));
        }
        
        return words;
    }
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
    std::string new_token = this->inv_vocab.at(pair.first) + this->inv_vocab.at(pair.second);
    
    // Add new token to vocabulary
    this->vocab[new_token] = new_token_id;
    this->inv_vocab[new_token_id] = new_token;
    this->merges[pair] = new_token_id;
    
    // Update word counts by replacing occurrences of the pair
    std::unordered_map<std::string, int> new_word_counts;
    
    for (const auto& [word, count] : word_counts) {
        std::string new_word;
        size_t pos = 0;
        
        while (pos < word.size()) {
            // Check if we found the pair at this position
            size_t first_len = this->inv_vocab.at(pair.first).size();
            size_t second_len = this->inv_vocab.at(pair.second).size();
            
            if (pos + first_len + second_len <= word.size() &&
                word.substr(pos, first_len) == this->inv_vocab.at(pair.first) &&
                word.substr(pos + first_len, second_len) == this->inv_vocab.at(pair.second)) {
                new_word += new_token;
                pos += first_len + second_len;
            } else {
                new_word += word[pos];
                pos++;
            }
        }
        
        new_word_counts[new_word] += count;
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
        if (!is_valid_utf8_impl(text.data(), text.size())) {
            std::cerr << "Warning: Invalid UTF-8 in training corpus: " << text << std::endl;
            // Skip invalid text
            continue;
        }
    }
    
    // Tokenize the entire corpus into token sequences with frequencies
    std::vector<std::pair<std::vector<TokenID>, int>> tokenized_corpus;
    std::unordered_map<std::vector<TokenID>, int, VectorHash> sequence_counts;
    
    // First, split text into words and tokenize each word
    for (const auto& text : corpus) {
        auto words = pimpl_->split_text(text);
        for (const auto& word : words) {
            // Convert word to initial token sequence (characters)
            auto tokens = pimpl_->word_to_token_ids(word);
            
            // Count frequency of this token sequence
            sequence_counts[tokens]++;
        }
    }
    
    // Convert to vector for easier processing
    tokenized_corpus.reserve(sequence_counts.size());
    for (const auto& [sequence, count] : sequence_counts) {
        tokenized_corpus.emplace_back(sequence, count);
    }
    
    // Clear the temporary map to save memory
    sequence_counts.clear();
    
    // BPE training algorithm with safety limit
    int iteration = 0;
    int max_iterations = 10000;
    
    // Pre-allocate pair counts
    std::unordered_map<std::pair<TokenID, TokenID>, int, PairHash> pair_counts;
    pair_counts.reserve(1000000); // Reserve space for 1M pairs
    
    while (pimpl_->vocab.size() < vocab_size && iteration < max_iterations) {
        // Count pairs in token sequences
        pair_counts.clear();
        pimpl_->get_pair_counts_from_sequences(tokenized_corpus, pair_counts);
        
        if (pair_counts.empty()) {
            std::cout << "No more pairs to merge. Stopping early." << std::endl;
            break;
        }
        
        // Find most frequent pair
        auto max_pair = std::max_element(
            pair_counts.begin(), pair_counts.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; }
        );
        
        // Debug output - show what we're merging
        if (pimpl_->debug_logging) {
            std::string first_str = pimpl_->inv_vocab.count(max_pair->first.first) ? 
                pimpl_->inv_vocab.at(max_pair->first.first) : "<?>";
            std::string second_str = pimpl_->inv_vocab.count(max_pair->first.second) ? 
                pimpl_->inv_vocab.at(max_pair->first.second) : "<?>";
            std::cout << "Iteration " << iteration 
                      << ": Merging '" << first_str << "' + '" << second_str 
                      << "' → count: " << max_pair->second << std::endl;
        }
        
        // Perform merge on token sequences
        pimpl_->perform_merge_on_sequences(max_pair->first, pimpl_->next_token_id, tokenized_corpus);
        pimpl_->next_token_id++;
        iteration++;
        
        // Periodically check memory usage and clean up
        if (iteration % 500 == 0) {
            size_t current_memory = get_peak_memory_usage();
            std::cout << "Memory after " << iteration << " iterations: " 
                      << (current_memory - start_memory) / (1024 * 1024) << "MB\n";
            std::cout << "Vocabulary size: " << pimpl_->vocab.size() << std::endl;
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
    
    pair_counts.clear();
    pair_counts.reserve(word_counts.size() * 10);
    
    for (const auto& [word, count] : word_counts) {
        // Tokenize the word using the current vocabulary
        auto tokens = word_to_token_ids(word);
        
        // Count pairs in the tokenized representation
        for (size_t i = 0; i < tokens.size() - 1; i++) {
            auto pair = std::make_pair(tokens[i], tokens[i+1]);
            pair_counts[pair] += count;
        }
    }
}

std::vector<TokenID> BPETokenizer::Impl::word_to_token_ids(const std::string& word) const {
    std::vector<TokenID> tokens;
    
    if (normalization_enabled) {
        // Use Unicode-aware splitting
        std::vector<std::string> characters;
        if (cache_enabled) {
            characters = unicode_cache.get_split(word);
        } else {
            characters = unicode::unicode_split(word);
        }
        
        for (const auto& character : characters) {
            if (auto it = vocab.find(character); it != vocab.end()) {
                tokens.push_back(it->second);
            } else if (byte_fallback_enabled) {
                // Fall back to byte encoding for unknown characters
                for (unsigned char c : character) {
                    std::string byte_str(1, static_cast<char>(c));
                    if (auto byte_it = vocab.find(byte_str); byte_it != vocab.end()) {
                        tokens.push_back(byte_it->second);
                    } else {
                        tokens.push_back(unknown_token_id);
                    }
                }
            } else {
                tokens.push_back(unknown_token_id);
            }
        }
    } else {
        // Non-Unicode mode: treat as ASCII
        for (char c : word) {
            std::string token(1, c);
            if (auto it = vocab.find(token); it != vocab.end()) {
                tokens.push_back(it->second);
            } else {
                tokens.push_back(unknown_token_id);
            }
        }
    }
    
    return tokens;
}

size_t BPETokenizer::vocab_size() const {
    return pimpl_->vocab.size();
}

std::vector<TokenID> BPETokenizer::encode(const std::string& text) const {
    pimpl_->log_encode_start(text);
    
    // Validate UTF-8 before processing
    if (!is_valid_utf8_impl(text.data(), text.size())) {
        if (pimpl_->byte_fallback_enabled) {
            return pimpl_->handle_invalid_utf8(text);
        } else {
            return {pimpl_->unknown_token_id};
        }
    }
    
    // Normalize the text first
    std::string normalized = pimpl_->normalization_enabled ? 
        pimpl_->unicode_cache.get_normalized(text) : text;
    
    // Split into words
    auto words = pimpl_->split_text(normalized);
    pimpl_->log_word_split(words);
    
    std::vector<TokenID> tokens;
    
    for (const auto& word : words) {
        // Convert word to initial tokens (characters)
        auto word_tokens = pimpl_->word_to_token_ids(word);
        pimpl_->log_word_tokens(word, word_tokens);
        
        // Apply BPE merges
        bool changed;
        do {
            changed = false;
            for (size_t i = 0; i < word_tokens.size() - 1; i++) {
                auto pair = std::make_pair(word_tokens[i], word_tokens[i+1]);
                if (auto it = pimpl_->merges.find(pair); it != pimpl_->merges.end()) {
                    // Replace the pair with the merged token
                    word_tokens[i] = it->second;
                    word_tokens.erase(word_tokens.begin() + i + 1);
                    changed = true;
                    pimpl_->log_merge_result(word_tokens);
                    // Restart from the beginning to catch new pairs
                    i = 0;
                }
            }
        } while (changed);
        
        tokens.insert(tokens.end(), word_tokens.begin(), word_tokens.end());
        
        // Add space between words (except after the last word)
        if (&word != &words.back() && pimpl_->vocab.find(" ") != pimpl_->vocab.end()) {
            tokens.push_back(pimpl_->vocab.at(" "));
        }
    }
    
    pimpl_->log_final_tokens(tokens);
    return tokens;
}

// Add debug logging to the decode method
std::string BPETokenizer::decode(const std::vector<TokenID>& tokens) const {
    pimpl_->log_decode_start(tokens);
    
    std::string text;
    text.reserve(tokens.size() * 3); // Estimate average token length
    
    for (TokenID token_id : tokens) {
        std::string token_text;
        if (pimpl_->inv_vocab.find(token_id) != pimpl_->inv_vocab.end()) {
            token_text = pimpl_->inv_vocab.at(token_id);
        } else {
            token_text = pimpl_->unknown_token;
        }
        pimpl_->log_token_decoding(token_id, token_text);
        text += token_text;
    }
    
    pimpl_->log_final_decoding(text);
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

} // namespace lm
