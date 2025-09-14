// include/lm/inference/inference_engine.hpp
#pragma once

#include <string>
#include <memory>
#include "../conversation/conversation_manager.hpp"
#include "lm/models/transformer_model.hpp"
#include "lm/tokenizer/bpe_tokenizer.hpp"
#include "lm/generation/sampler.hpp"

namespace lm {

class InferenceEngine {
public:
    InferenceEngine(const std::string& model_path, 
                  const std::string& tokenizer_path,
                  const std::string& sampler_type = "top_p",
                  float temperature = 0.8);
    
    std::string chat(const std::string& input, 
                   size_t max_length = 100,
                   float temperature = 0.8);
    
    void reset_conversation();
    void save_conversation(const std::string& path) const;
    void load_conversation(const std::string& path);
    
private:
    std::shared_ptr<TransformerModel> model_;
    std::shared_ptr<BPETokenizer> tokenizer_;
    std::unique_ptr<ConversationManager> conversation_manager_;
    
    void load_model(const std::string& path);
    void load_tokenizer(const std::string& path);
    std::unique_ptr<Sampler> create_sampler(const std::string& type, float temperature);
};

} // namespace lm

