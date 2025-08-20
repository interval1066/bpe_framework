/* BPE Tokenizer Implementation File

Here's the complete `src/tokenizer/bpe_tokenizer.cpp` file:

```cpp*/
#include "lm/tokenizer/bpe_tokenizer.hpp"
#include "lm/tokenizer/unicode_utils.hpp"
#include <fstream>
#include <sstream>
#include <queue>
#include <algorithm>
#include <stdexcept>

namespace lm {

struct BPETokenizer::Impl {
    std::unordered_map<std::string, int> vocab;
    std::unordered_map<int, std::string> inv_vocab;
    std::map<std::pair<int, int>, int> merges;
    std::unordered_map<std::string, int> special_tokens;
    std::string unknown_token = "<unk>";
    int unknown_token_id = -1;
    int next_token_id = 0;
    bool normalization_enabled = true;
    bool byte_fallback_enabled = true;
    
    // Helper functions
    std::vector<std::string> split_text(const std::string& text) const;
    std::vector<int> word_to_token_ids(const std::string& word) const;
    void initialize_vocab();
    void count_word_frequencies(const std::vector<std::string>& words,
                               std::unordered_map<std::string, int>& word_counts) const;
    void get_pair_counts(const std::unordered_map<std::string, int>& word_counts,
                        std::map<std::pair<int, int>, int>& pair_counts) const;
    void perform_merge(const std::pair<int, int>& pair, int new_token_id,
                      std::unordered_map<std::string, int>& word_counts);
};

BPETokenizer::BPETokenizer() : pimpl_(new Impl) {
    pimpl_->initialize_vocab();
}

BPETokenizer::~BPETokenizer() = default;

/*==========================
void BPETokenizer::Impl::initialize_vocab() {
    // Add basic bytes to vocabulary
    for (int i = 0; i < 256; i++) {
        std::string token(1, static_cast<char>(i));
        vocab[token] = next_token_id;
        inv_vocab[next_token_id] = token;
        next_token_id++;
    }
    
    // Add special tokens
    add_special_token(unknown_token);
    unknown_token_id = vocab[unknown_token];  // This line should set unknown_token_id
}

=========================*/
void BPETokenizer::Impl::initialize_vocab() {
    // Add basic bytes to vocabulary
    for (int i = 0; i < 256; i++) {
        std::string token(1, static_cast<char>(i));
        vocab[token] = next_token_id;
        inv_vocab[next_token_id] = token;
        next_token_id++;
    }
    
    // Add special tokens directly instead of calling add_special_token
    if (vocab.find(unknown_token) == vocab.end()) {
        vocab[unknown_token] = next_token_id;
        inv_vocab[next_token_id] = unknown_token;
        special_tokens[unknown_token] = next_token_id;
        unknown_token_id = next_token_id;
        next_token_id++;
    }
}

//=======================
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

std::vector<int> BPETokenizer::Impl::word_to_token_ids(const std::string& word) const {
    std::vector<int> tokens;
    
    if (normalization_enabled) {
        std::string normalized = unicode::normalize(word);
        auto characters = unicode::split_on_character_boundaries(normalized);
        
        for (const auto& character : characters) {
            if (vocab.find(character) != vocab.end()) {
                tokens.push_back(vocab.at(character));  // This pushes an int (token ID)
            } else if (byte_fallback_enabled) {
                // If character not found, try to split into bytes
                for (unsigned char c : character) {
                    std::string byte_str(1, static_cast<char>(c));
                    if (vocab.find(byte_str) != vocab.end()) {
                        tokens.push_back(vocab.at(byte_str));  // This pushes an int
                    } else {
                        tokens.push_back(unknown_token_id);  // This should push an int
                    }
                }
            } else {
                tokens.push_back(unknown_token_id);  // This should push an int
            }
        }
    } else {
        // Non-Unicode mode: treat as ASCII
        for (char c : word) {
            std::string token(1, c);
            if (vocab.find(token) != vocab.end()) {
                tokens.push_back(vocab.at(token));  // This pushes an int
            } else {
                tokens.push_back(unknown_token_id);  // This should push an int
            }
        }
    }
    
    return tokens;
}

void BPETokenizer::Impl::count_word_frequencies(const std::vector<std::string>& words,
                                               std::unordered_map<std::string, int>& word_counts) const {
    for (const auto& word : words) {
        word_counts[word]++;
    }
}

void BPETokenizer::Impl::get_pair_counts(const std::unordered_map<std::string, int>& word_counts,
                                        std::map<std::pair<int, int>, int>& pair_counts) const {
    for (const auto& [word, count] : word_counts) {
        auto tokens = word_to_token_ids(word);
        for (size_t i = 0; i < tokens.size() - 1; i++) {
            auto pair = std::make_pair(tokens[i], tokens[i+1]);
            pair_counts[pair] += count;
        }
    }
}

void BPETokenizer::Impl::perform_merge(const std::pair<int, int>& pair, int new_token_id,
                                      std::unordered_map<std::string, int>& word_counts) {
    std::string new_token = inv_vocab.at(pair.first) + inv_vocab.at(pair.second);
    
    // Add new token to vocabulary
    vocab[new_token] = new_token_id;
    inv_vocab[new_token_id] = new_token;
    
    // Record the merge
    merges[pair] = new_token_id;
    
    // Update word counts with new merges
    std::unordered_map<std::string, int> new_word_counts;
    for (const auto& [word, count] : word_counts) {
        std::string new_word = word;
        size_t pos = 0;
        while ((pos = new_word.find(inv_vocab.at(pair.first) + inv_vocab.at(pair.second), pos)) != std::string::npos) {
            new_word.replace(pos, 2, new_token);
            pos += new_token.length();
        }
        new_word_counts[new_word] += count;
    }
    
    word_counts = std::move(new_word_counts);
}

void BPETokenizer::train(const std::vector<std::string>& corpus, size_t vocab_size) {
    if (corpus.empty()) {
        throw std::invalid_argument("Corpus cannot be empty");
    }
    
    // Split text into words
    std::vector<std::string> words;
    for (const auto& text : corpus) {
        auto text_words = pimpl_->split_text(text);
        words.insert(words.end(), text_words.begin(), text_words.end());
    }
    
    // Count word frequencies
    std::unordered_map<std::string, int> word_counts;
    pimpl_->count_word_frequencies(words, word_counts);
    
    // BPE training algorithm
    while (pimpl_->vocab.size() < vocab_size) {
        // Count pairs
        std::map<std::pair<int, int>, int> pair_counts;
        pimpl_->get_pair_counts(word_counts, pair_counts);
        
        if (pair_counts.empty()) {
            break; // No more pairs to merge
        }
        
        // Find most frequent pair
        auto max_pair = std::max_element(
            pair_counts.begin(), pair_counts.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; }
        );
        
        // Perform merge
        pimpl_->perform_merge(max_pair->first, pimpl_->next_token_id, word_counts);
        pimpl_->next_token_id++;
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

std::vector<int> BPETokenizer::encode(const std::string& text) const {
    auto words = pimpl_->split_text(text);
    std::vector<int> tokens;
    
    for (const auto& word : words) {
        // Start with character-level tokens
        auto word_tokens = pimpl_->word_to_token_ids(word);
        
        // Apply BPE merges
        bool changed;
        do {
            changed = false;
            for (const auto& [pair, merged_id] : pimpl_->merges) {
                for (size_t i = 0; i < word_tokens.size() - 1; i++) {
                    if (word_tokens[i] == pair.first && word_tokens[i+1] == pair.second) {
                        word_tokens[i] = merged_id;
                        word_tokens.erase(word_tokens.begin() + i + 1);
                        changed = true;
                        break;
                    }
                }
                if (changed) break;
            }
        } while (changed);
        
        tokens.insert(tokens.end(), word_tokens.begin(), word_tokens.end());
        
        // Add space token between words (except after last word)
        if (&word != &words.back()) {
            tokens.push_back(pimpl_->vocab.at(" "));
        }
    }
    
    return tokens;
}

std::string BPETokenizer::decode(const std::vector<int>& tokens) const {
    std::string text;
    for (int token_id : tokens) {
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
        int id;
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
        int first, second, new_id;
        file >> first >> second >> new_id;
        pimpl_->merges[{first, second}] = new_id;
    }
    
    return true;
}

size_t BPETokenizer::vocab_size() const {
    return pimpl_->vocab.size();
}

std::string BPETokenizer::id_to_token(int id) const {
    if (pimpl_->inv_vocab.find(id) != pimpl_->inv_vocab.end()) {
        return pimpl_->inv_vocab.at(id);
    }
    return pimpl_->unknown_token;
}

int BPETokenizer::token_to_id(const std::string& token) const {
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

/*void BPETokenizer::add_special_token(const std::string& token) {
    if (pimpl_->vocab.find(token) == pimpl_->vocab.end()) {
        pimpl_->vocab[token] = pimpl_->next_token_id;
        pimpl_->inv_vocab[pimpl_->next_token_id] = token;
        pimpl_->special_tokens[token] = pimpl_->next_token_id;
        pimpl_->next_token_id++;
    }
}*/

void BPETokenizer::set_normalization(bool enabled) {
    pimpl_->normalization_enabled = enabled;
}

void BPETokenizer::set_byte_fallback(bool enabled) {
    pimpl_->byte_fallback_enabled = enabled;
}

void BPETokenizer::add_special_token(const std::string& token) {
    if (pimpl_->vocab.find(token) == pimpl_->vocab.end()) {
        pimpl_->vocab[token] = pimpl_->next_token_id;
        pimpl_->inv_vocab[pimpl_->next_token_id] = token;
        pimpl_->special_tokens[token] = pimpl_->next_token_id;
        pimpl_->next_token_id++;
    }
}

} // namespace lm
/*```

This implementation provides a complete BPE tokenizer with Unicode support, featuring:

1. **Unicode-aware text processing** with normalization
2. **Byte-level fallback** for unknown Unicode characters
3. **Configurable normalization** and fallback behavior
4. **Complete BPE training algorithm** with frequency-based merging
5. **Serialization/deserialization** for saving and loading trained models
6. **Special token handling** for custom tokens like `<unk>`
7. **Vocabulary management** with bidirectional token-ID mapping

The implementation uses the PIMPL idiom to hide implementation details while providing a clean public interface. It integrates with the Unicode utilities to properly handle multilingual text while maintaining backward compatibility with ASCII text.*/
