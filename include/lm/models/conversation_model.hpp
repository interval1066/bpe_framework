#pragma once

#include "lm/models/transformer_model.hpp"
#include "bpe_tokenizer.hpp"
#include "context_manager.hpp"
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <iomanip>

namespace lm {

    class ConversationModel {
    public:
        ConversationModel(size_t vocab_size, 
                     size_t d_model = 512, 
                     size_t n_layers = 6, 
                     size_t n_heads = 8,
                     size_t d_ff = 2048,
                     float dropout = 0.1);
    
        // Train the model
        void train(const std::vector<std::string>& conversations);
    
        // Generate a response with context management
        std::string generate_response(const std::string& user_input);
    
        // Context management
        void clear_context();
        void set_system_prompt(const std::string& prompt);
        size_t get_context_token_count() const;
    
        // Save and load
        bool save_model(const std::string& path);
        bool load_model(const std::string& path);
    
        // Set tokenizer
        void set_tokenizer(std::shared_ptr<BPETokenizer> tokenizer) { 
           tokenizer_ = tokenizer; 
            context_manager_ = std::make_unique<ContextManager>(2048, 20);
        }

        inline size_t vocab_size() const {
            return transformer_->vocab_size();
        }

        void set_verbose(bool verbose) { verbose_ = verbose; }
        void set_max_response_length(size_t length) { max_response_length_ = length; }
        void set_temperature(float temperature) { temperature_ = temperature; }

        bool save_checkpoint(const std::string& path);
        bool load_checkpoint(const std::string& path);
        void serialize(std::ostream& stream) const;

        void deserialize(std::istream& stream);

    private:
        std::shared_ptr<BPETokenizer> tokenizer_;
        std::unique_ptr<TransformerModel> transformer_;
        std::unique_ptr<ContextManager> context_manager_;

        std::string system_prompt_;
        TokenID pad_token_id_;
        bool verbose_ = false;

        size_t max_response_length_ = 20;
        float temperature_ = 0.8f;    
        // Format conversation for training
        std::string format_conversation(const std::vector<std::string>& turns);

        size_t training_epochs_ = 0;
        float best_loss_ = std::numeric_limits<float>::max();
        std::string training_timestamp_;

        // Helper methods
        std::string get_current_timestamp() const {
           auto now = std::chrono::system_clock::now();
           auto in_time_t = std::chrono::system_clock::to_time_t(now);
        
            std::stringstream ss;
            ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %X");
            return ss.str();
        }
    };

} // namespace lm

