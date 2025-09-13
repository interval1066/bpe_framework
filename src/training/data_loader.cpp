// src/training/data_loader.cpp
#include "data_loader.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <random>
#include <algorithm>

namespace lm {

ConversationDataLoader::ConversationDataLoader(const std::string& file_path, 
                                             BPETokenizer& tokenizer,
                                             size_t batch_size, 
                                             size_t seq_length)
    : tokenizer_(tokenizer), batch_size_(batch_size), seq_length_(seq_length), 
      current_index_(0) {
    load_conversations(file_path);
}

void ConversationDataLoader::load_conversations(const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open conversation data file: " + file_path);
    }
    
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            auto tokens = tokenize_conversation(line);
            if (!tokens.empty()) {
                conversations_.push_back(tokens);
            }
        }
    }
    
    if (conversations_.empty()) {
        throw std::runtime_error("No conversations loaded from file: " + file_path);
    }
    
    // Shuffle conversations for better training
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(conversations_.begin(), conversations_.end(), g);
    
    std::cout << "Loaded " << conversations_.size() << " conversations" << std::endl;
}

std::vector<int> ConversationDataLoader::tokenize_conversation(const std::string& conversation) {
    // Simple conversation format: User: Hello|AI: Hi there|User: How are you?
    // We'll split by | and tokenize each part
    
    std::vector<int> all_tokens;
    std::stringstream ss(conversation);
    std::string part;
    
    while (std::getline(ss, part, '|')) {
        if (!part.empty()) {
            auto tokens = tokenizer_.encode(part);
            all_tokens.insert(all_tokens.end(), tokens.begin(), tokens.end());
            
            // Add separator token (assuming 3 is SEP)
            all_tokens.push_back(3);
        }
    }
    
    // Remove the last separator if present
    if (!all_tokens.empty() && all_tokens.back() == 3) {
        all_tokens.pop_back();
    }
    
    return all_tokens;
}

bool ConversationDataLoader::has_next() const {
    return current_index_ < conversations_.size();
}

std::pair<Tensor, Tensor> ConversationDataLoader::next_batch() {
    if (!has_next()) {
        throw std::out_of_range("No more batches available");
    }
    
    size_t end_index = std::min(current_index_ + batch_size_, conversations_.size());
    size_t actual_batch_size = end_index - current_index_;
    
    // Find the maximum sequence length in this batch
    size_t max_seq_len = 0;
    for (size_t i = current_index_; i < end_index; i++) {
        max_seq_len = std::max(max_seq_len, conversations_[i].size());
    }
    
    // Limit to the configured sequence length and add 1 for targets
    max_seq_len = std::min(max_seq_len, seq_length_);
    
    // Create input and target tensors
    Tensor inputs({actual_batch_size, max_seq_len}, false);
    Tensor targets({actual_batch_size, max_seq_len}, false);
    
    // Fill the tensors with data
    for (size_t i = 0; i < actual_batch_size; i++) {
        const auto& tokens = conversations_[current_index_ + i];
        size_t seq_len = std::min(tokens.size(), max_seq_len);
        
        for (size_t j = 0; j < seq_len; j++) {
            inputs(i, j) = static_cast<float>(tokens[j]);
            
            // For language modeling, target is the next token
            if (j < seq_len - 1) {
                targets(i, j) = static_cast<float>(tokens[j + 1]);
            } else {
                targets(i, j) = -100.0f; // Standard value for ignored indices in loss
            }
        }
        
        // Pad the rest of the sequence if needed
        for (size_t j = seq_len; j < max_seq_len; j++) {
            inputs(i, j) = 0.0f; // Pad token ID (assuming 0 is pad)
            targets(i, j) = -100.0f; // Ignore in loss
        }
    }
    
    current_index_ = end_index;
    return {inputs, targets};
}

void ConversationDataLoader::reset() {
    current_index_ = 0;
    
    // Reshuffle for the next epoch
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(conversations_.begin(), conversations_.end(), g);
}

size_t ConversationDataLoader::num_batches() const {
    return (conversations_.size() + batch_size_ - 1) / batch_size_;
}

} // namespace lm

