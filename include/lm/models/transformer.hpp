// include/lm/models/transformer.hpp
#pragma once

#include "language_model.hpp"
#include "transformer_block.hpp"
#include "attention.hpp"
#include "feed_forward.hpp"
#include "layer_norm.hpp"
#include <vector>
#include <memory>

namespace lm {

    class Transformer : public LanguageModel {
        size_t vocab_size_,
            d_model_,
            num_heads_,
            d_ff_,
            num_layers_,
            max_seq_length_;
        float dropout_;
    
        // Embedding layers
        Tensor token_embeddings_;
        Tensor positional_embeddings_;
    
        // Transformer blocks
        std::vector<TransformerBlock> layers_;

        // Final layer norm and output projection
        LayerNorm final_layer_norm_;
        Tensor output_projection_;
    
    public:
        Transformer(size_t vocab_size, size_t d_model, size_t num_heads, 
               size_t d_ff, size_t num_layers, float dropout, size_t max_seq_length);
    
        // LanguageModel interface implementation
        std::vector<Tensor> get_parameters() const override;
        void set_parameters(const std::vector<Tensor>& params) override;
        Tensor forward(const std::vector<TokenID>& input) override;
        Tensor forward(const std::vector<TokenID>& input, 
                  const std::vector<TokenID>& targets) override;
    
        size_t get_vocab_size() const override { return vocab_size_; }
        size_t get_max_sequence_length() const override { return max_seq_length_; }
    
        void save(const std::string& path) const override;
        void load(const std::string& path) override;
    
        // Additional methods specific to Transformer
        Tensor embed(const std::vector<TokenID>& tokens) const;
        Tensor apply_positional_encoding(const Tensor& embeddings, size_t seq_length) const;
    };

} // namespace lm

