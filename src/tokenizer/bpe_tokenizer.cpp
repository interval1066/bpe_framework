#include "lm/tokenizer/bpe_tokenizer.hpp"
#include "lm/tokenizer/unicode_utils.hpp"
#include <fstream>
#include <sstream>
#include <queue>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <sys/resource.h>

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
    
    // CPU Optimization: Batch processing
    void process_string_batch(const std::vector<std::string>& batch);
};

BPETokenizer::BPETokenizer() : pimpl_(new Impl) {
    pimpl_->initialize_vocab();
}

BPETokenizer::~BPETokenizer() = default;

void BPETokenizer::Impl::initialize_vocab() {
    vocab.reserve(65536);  // More realistic initial allocation
    inv_vocab.reserve(65536);
    
    // Add bytes with optimized insertion
    for (int i = 0; i < 256; i++) {
        std::string token(1, static_cast<char>(i));
        vocab.emplace(token, next_token_id);
        inv_vocab.emplace(next_token_id++, std::move(token));
    }
    
    // Add unknown token
    auto [it, inserted] = vocab.emplace(unknown_token, next_token_id);
    if (inserted) {
        inv_vocab.emplace(next_token_id, unknown_token);
        special_tokens.emplace(unknown_token, next_token_id);
        unknown_token_id = next_token_id++;
    } else {
        unknown_token_id = it->second;
    }
}

std::vector<std::string> BPETokenizer::Impl::split_text(const std::string& text) const {
    if (normalization_enabled) {
        std::string normalized = unicode::normalize(text);
        return unicode::unicode_split(normalized);
    } else {
        std::vector<std::string> words;
        std::istringstream iss(text);
        std::string word;
        
        while (iss >> word) {
            words.push_back(word);
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
    
    if (local_normalization) {
        std::string normalized = unicode::normalize(word);
        auto characters = unicode::split_on_character_boundaries(normalized);
        
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
    
    // Use a local reference to avoid repeated map lookups
    const auto& local_vocab = vocab;
    
    for (const auto& word : words) {
        // Use string_view to avoid copies where possible
        auto it = word_counts.find(word);
        if (it != word_counts.end()) {
            it->second++;
        } else {
            word_counts[word] = 1;
        }
    }
}

/*void BPETokenizer::Impl::perform_merge(const std::pair<TokenID, TokenID>& pair, TokenID new_token_id,
                                      std::unordered_map<std::string, int>& word_counts) {
    std::string new_token = inv_vocab.at(pair.first) + inv_vocab.at(pair.second);
    
    // Use move semantics to avoid copies
    auto shared_token = string_pool.intern(std::move(new_token));
    
    // Add new token to vocabulary
    vocab[*shared_token] = new_token_id;
    inv_vocab[new_token_id] = *shared_token;
    merges[pair] = new_token_id;
    
    // Update word counts more efficiently
    std::vector<std::string> to_remove;
    std::unordered_map<std::string, int> new_counts;
    
    for (auto& [word, count] : word_counts) {
        std::string new_word;
        new_word.reserve(word.size()); // Pre-allocate
        
        size_t pos = 0;
        while (pos < word.size()) {
            if (pos < word.size() - 1 && 
                word.substr(pos, inv_vocab.at(pair.first).size()) == inv_vocab.at(pair.first) &&
                word.substr(pos + inv_vocab.at(pair.first).size(), inv_vocab.at(pair.second).size()) == inv_vocab.at(pair.second)) {
                new_word += *shared_token;
                pos += inv_vocab.at(pair.first).size() + inv_vocab.at(pair.second).size();
            } else {
                new_word += word[pos++];
            }
        }
        
        if (new_word != word) {
            to_remove.push_back(word);
            new_counts[new_word] += count;
        }
    }
    
    // Remove old words and add new ones
    for (const auto& word : to_remove) {
        word_counts.erase(word);
    }
    for (auto& [word, count] : new_counts) {
        word_counts[word] += count;
    }
}*/

void BPETokenizer::Impl::perform_merge(const std::pair<TokenID, TokenID>& pair, TokenID new_token_id,
                                      std::unordered_map<std::string, int>& word_counts) {
    std::string new_token = inv_vocab.at(pair.first) + inv_vocab.at(pair.second);
    auto shared_token = string_pool.intern(new_token);
    
    // Add new token to vocabulary
    vocab[*shared_token] = new_token_id;
    inv_vocab[new_token_id] = *shared_token;
    
    // Record the merge
    merges[pair] = new_token_id;
    
    // Update word counts with new merges
    std::unordered_map<std::string, int> new_word_counts;
    for (const auto& [word, count] : word_counts) {
        std::string new_word = word;
        size_t pos = 0;
        while ((pos = new_word.find(inv_vocab.at(pair.first) + inv_vocab.at(pair.second), pos)) != std::string::npos) {
            new_word.replace(pos, 2, *shared_token);
            pos += shared_token->length();
        }
        new_word_counts[new_word] += count;
    }
    
    word_counts = std::move(new_word_counts);
}

// CPU Optimization: Batch processing for better cache utilization
void BPETokenizer::Impl::process_string_batch(const std::vector<std::string>& batch) {
    const size_t batch_size = batch.size();
    
    #ifdef __SSE4_2__
    // Predefined patterns to detect
    const std::string url_pattern = "http";
    const std::string email_pattern = "@";
    const std::string number_pattern = "0123456789";
    
    // Precompute pattern lengths
    const int url_len = url_pattern.size();
    const int email_len = email_pattern.size();
    const int number_len = number_pattern.size();
    
    // Load patterns into SSE registers
    __m128i url_pattern_vec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(url_pattern.c_str()));
    __m128i email_pattern_vec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(email_pattern.c_str()));
    __m128i number_pattern_vec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(number_pattern.c_str()));
    
    for (size_t i = 0; i < batch_size; i++) {
        const std::string& str = batch[i];
        const size_t len = str.size();
        const char* data = str.data();
        
        int url_count = 0;
        int email_count = 0;
        int number_count = 0;
        
        // Process 16 characters at a time
        size_t j = 0;
        for (; j + 16 <= len; j += 16) {
            __m128i chunk = _mm_loadu_si128(reinterpret_cast<const __m128i*>(data + j));
            
            // Search for URL pattern using SSE4.2 string instructions
            int url_idx = _mm_cmpestri(
                url_pattern_vec, url_len,
                chunk, 16,
                _SIDD_CMP_EQUAL_ORDERED | _SIDD_UBYTE_OPS
            );
            
            if (url_idx < 16) {
                url_count++;
                // Optional: Add special URL token if not already in vocabulary
                auto it = vocab.find("<URL>");
                if (it == vocab.end()) {
                    vocab["<URL>"] = next_token_id;
                    inv_vocab[next_token_id] = "<URL>";
                    next_token_id++;
                }
            }
            
            // Search for email pattern
            int email_idx = _mm_cmpestri(
                email_pattern_vec, email_len,
                chunk, 16,
                _SIDD_CMP_EQUAL_ORDERED | _SIDD_UBYTE_OPS
            );
            
            if (email_idx < 16) {
                email_count++;
                // Optional: Add special email token
                auto it = vocab.find("<EMAIL>");
                if (it == vocab.end()) {
                    vocab["<EMAIL>"] = next_token_id;
                    inv_vocab[next_token_id] = "<EMAIL>";
                    next_token_id++;
                }
            }
            
            // Search for number patterns using range comparison
            __m128i digit_lo = _mm_cmpgt_epi8(chunk, _mm_set1_epi8('0' - 1));
            __m128i digit_hi = _mm_cmplt_epi8(chunk, _mm_set1_epi8('9' + 1));
            __m128i digit_mask = _mm_and_si128(digit_lo, digit_hi);
            
            int digit_count = _mm_popcnt_u32(_mm_movemask_epi8(digit_mask));
            if (digit_count >= 3) { // Threshold for considering it a number
                number_count++;
                // Optional: Add special number token
                auto it = vocab.find("<NUMBER>");
                if (it == vocab.end()) {
                    vocab["<NUMBER>"] = next_token_id;
                    inv_vocab[next_token_id] = "<NUMBER>";
                    next_token_id++;
                }
            }
            
            // Additional pattern detection can be added here
        }
        
        // Process remaining characters with traditional methods
        for (; j < len; j++) {
            // Check for URL pattern
            if (j + url_len <= len && 
                strncmp(data + j, url_pattern.c_str(), url_len) == 0) {
                url_count++;
                j += url_len - 1; // Skip ahead
            }
            
            // Check for email pattern
            if (j + email_len <= len && 
                strncmp(data + j, email_pattern.c_str(), email_len) == 0) {
                email_count++;
            }
            
            // Check for number pattern
            if (data[j] >= '0' && data[j] <= '9') {
                number_count++;
            }
        }
        
        // Use pattern counts to make tokenization decisions
        if (url_count > 0) {
            // Apply special tokenization for URLs
        }
        
        if (email_count > 0) {
            // Apply special tokenization for emails
        }
        
        if (number_count > 5) { // Threshold for considering as numeric content
            // Apply special tokenization for numbers
        }
    }
    #else
    // Fallback implementation without SSE
    const std::string url_pattern = "http";
    const std::string email_pattern = "@";
    
    for (const auto& str : batch) {
        int url_count = 0;
        int email_count = 0;
        int number_count = 0;
        
        // Simple pattern matching without SSE
        for (size_t j = 0; j < str.size(); j++) {
            // Check for URL pattern
            if (j + url_pattern.size() <= str.size() && 
                str.substr(j, url_pattern.size()) == url_pattern) {
                url_count++;
                j += url_pattern.size() - 1;
            }
            
            // Check for email pattern
            if (j + email_pattern.size() <= str.size() && 
                str.substr(j, email_pattern.size()) == email_pattern) {
                email_count++;
            }
            
            // Check for numbers
            if (str[j] >= '0' && str[j] <= '9') {
                number_count++;
            }
        }
        
        // Same logic as above for tokenization decisions
        if (url_count > 0) {
            auto it = vocab.find("<URL>");
            if (it == vocab.end()) {
                vocab["<URL>"] = next_token_id;
                inv_vocab[next_token_id] = "<URL>";
                next_token_id++;
            }
        }
        
        // Similar handling for other patterns
    }
    #endif
}

void BPETokenizer::train(const std::vector<std::string>& corpus, size_t vocab_size) {
    size_t start_memory = get_peak_memory_usage();
    
    if (corpus.empty()) {
        throw std::invalid_argument("Corpus cannot be empty");
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
    word_counts.reserve(words.size());
    
    pimpl_->count_word_frequencies(words, word_counts);
    
    // BPE training algorithm with safety limit
    int iteration = 0;
    int max_iterations = 10000;
    
    while (pimpl_->vocab.size() < vocab_size && iteration < max_iterations) {
        // Count pairs
        std::unordered_map<std::pair<TokenID, TokenID>, int, PairHash> pair_counts;
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
        
        // Periodically check memory usage
        if (iteration % 500 == 0) {
            size_t current_memory = get_peak_memory_usage();
            std::cout << "Memory after " << iteration << " iterations: " 
                      << (current_memory - start_memory) / (1024 * 1024) << "MB\n";
        }
    }
    
    if (iteration >= max_iterations) {
        std::cout << "Reached maximum iterations. Stopping training." << std::endl;
    }
    
    size_t end_memory = get_peak_memory_usage();
    std::cout << "Training completed in " << iteration << " iterations\n";
    std::cout << "Peak memory used: " << (end_memory - start_memory) / (1024 * 1024) << "MB\n";
    std::cout << "Final vocabulary size: " << pimpl_->vocab.size() << std::endl;

    // Add periodic memory cleanup
    if (iteration % 1000 == 0) {
        pimpl_->string_pool.clear();
    }
}

void BPETokenizer::train_from_file(const std::string& filename, size_t vocab_size) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    std::vector<std::string> corpus;
    std::string line;
    while (std::getline(file, line)) {
        corpus.push_back(line);
    }
    
    train(corpus, vocab_size);
}

std::vector<TokenID> BPETokenizer::encode(const std::string& text) const {
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

size_t BPETokenizer::vocab_size() const {
    return pimpl_->vocab.size();
}

std::string BPETokenizer::id_to_token(TokenID id) const {
    if (pimpl_->inv_vocab.find(id) != pimpl_->inv_vocab.end()) {
        return pimpl_->inv_vocab.at(id);
    }
    return pimpl_->unknown_token;
}

TokenID BPETokenizer::token_to_id(const std::string& token) const {
    if (pimpl_->vocab.find(token) != pimpl_->vocab.end()) {
        return pimpl_->vocab.at(token);
    }
    return pimpl_->unknown_token_id;
}

void BPETokenizer::set_unknown_token(const std::string& token) {
    pimpl_->unknown_token = token;
    // If token already exists in vocab, use its ID
    if (pimpl_->vocab.find(token) != pimpl_->vocab.end()) {
        pimpl_->unknown_token_id = pimpl_->vocab.at(token);
    } else {
        // Add to vocab
        pimpl_->vocab[token] = pimpl_->next_token_id;
        pimpl_->inv_vocab[pimpl_->next_token_id] = token;
        pimpl_->unknown_token_id = pimpl_->next_token_id;
        pimpl_->next_token_id++;
    }
}

void BPETokenizer::add_special_token(const std::string& token) {
    if (pimpl_->vocab.find(token) == pimpl_->vocab.end()) {
        pimpl_->vocab[token] = pimpl_->next_token_id;
        pimpl_->inv_vocab[pimpl_->next_token_id] = token;
        pimpl_->special_tokens[token] = pimpl_->next_token_id;
        pimpl_->next_token_id++;
    }
}

void BPETokenizer::set_normalization(bool enabled) {
    pimpl_->normalization_enabled = enabled;
}

void BPETokenizer::set_byte_fallback(bool enabled) {
    pimpl_->byte_fallback_enabled = enabled;
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

Eigen::VectorXi BPETokenizer::encode_to_vector(const std::string& text) const {
    auto token_ids = encode(text);
    Eigen::VectorXi result(token_ids.size());
    
    for (size_t i = 0; i < token_ids.size(); ++i) {
        result(i) = static_cast<int>(token_ids[i]);
    }
    
    return result;
}

std::string BPETokenizer::decode_from_vector(const Eigen::VectorXi& tokens) const {
    std::vector<TokenID> token_ids(tokens.size());
    
    for (int i = 0; i < tokens.size(); ++i) {
        token_ids[i] = static_cast<TokenID>(tokens(i));
    }
    
    return decode(token_ids);
}

Eigen::VectorXf BPETokenizer::token_frequencies() const {
    // This is a placeholder implementation
    // In a real implementation, you would track token frequencies during training
    size_t vocab_size = vocab_size;
    Eigen::VectorXf frequencies(vocab_size);
    
    // Initialize with equal frequencies (placeholder)
    for (size_t i = 0; i < vocab_size; ++i) {
        frequencies(i) = 1.0f / vocab_size;
    }
    
    return frequencies;
}

} // namespace lm
