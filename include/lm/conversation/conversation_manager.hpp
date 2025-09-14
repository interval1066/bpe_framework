// include/lm/conversation/conversation_manager.hpp
#pragma once

#include <string>
#include <vector>
#include <memory>
#include "../models/transformer_model.hpp"
#include "../tokenizer/bpe_tokenizer.hpp"
#include "../generation/sampler.hpp"

namespace lm {

class ConversationManager {
public:
    ConversationManager(std::shared_ptr<TransformerModel> model,
                      std::shared_ptr<BPETokenizer> tokenizer,
                      std::unique_ptr<Sampler> sampler);
    
    std::string generate_response(const std::string& input, 
                                size_t max_length = 100,
                                float temperature = 0.8f);
    
    void add_to_history(const std::string& utterance, bool is_user = true);
    void clear_history();
    std::vector<std::string> get_history() const;
    
    void set_sampler(std::unique_ptr<Sampler> sampler);
    
    // Add these methods for serialization
    void save_conversation(const std::string& path) const;  // Note the const qualifier
    void load_conversation(const std::string& path);
    
private:
    std::shared_ptr<TransformerModel> model_;
    std::shared_ptr<BPETokenizer> tokenizer_;
    std::unique_ptr<Sampler> sampler_;
    std::vector<std::string> conversation_history_;
    size_t max_history_length_;
    
    std::vector<int> format_conversation() const;
};

} // namespace lm

