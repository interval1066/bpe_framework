// include/lm/training/data_loader.hpp
#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <random>
#include "../core/tensor.hpp"
#include "../tokenizer/bpe_tokenizer.hpp"

namespace lm {

class ConversationDataLoader {
public:
    ConversationDataLoader(const std::string& file_path, BPETokenizer& tokenizer, 
                         size_t batch_size, size_t seq_length);
    
    bool has_next() const;
    std::pair<Tensor, Tensor> next_batch(); // Returns (input, target) tensors
    
    void reset();
    size_t num_batches() const;

private:
    BPETokenizer& tokenizer_;
    size_t batch_size_;
    size_t seq_length_;
    std::vector<std::vector<int>> conversations_;
    size_t current_index_;
    
    void load_conversations(const std::string& file_path);
    std::vector<int> tokenize_conversation(const std::string& conversation);
};

} // namespace lm

