#pragma once

#include <vector>
#include <memory>
#include <iostream>
#include <fstream>
#include "../core/tensor.hpp"
#include <cereal/types/vector.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/archives/binary.hpp>

namespace lm {

using TokenID = unsigned int;

class TransformerModel {
public:
    // Factory method instead of constructor
    static std::shared_ptr<TransformerModel> create();
    static std::shared_ptr<TransformerModel> create(size_t vocab_size, size_t d_model, 
                                                  size_t n_layers, size_t n_heads, 
                                                  size_t d_ff, float dropout);
    
    // Public interface
    std::vector<float> forward(const std::vector<TokenID>& input_tokens);
    void backward(const std::vector<float>& grad_output);
    float calculate_loss(const std::vector<float>& logits, 
                       const std::vector<TokenID>& targets);

    float train_step(const std::vector<TokenID>& input_tokens, 
                    const std::vector<TokenID>& target_tokens,
                    float learning_rate = 0.01f,
                    float max_grad_norm = 1.0f);  // 4 parameters total

    std::vector<TokenID> generate(const std::vector<TokenID>& context, 
                                size_t max_length, float temperature);
    std::vector<Tensor*> parameters();
    void zero_grad();
    float clip_gradients(float max_norm = 1.0f);
    void serialize(std::ostream& stream) const;
    void deserialize(std::istream& stream);
    bool save(const std::string& filename);
    bool load(const std::string& filename);
    void clip_gradients_simple(float max_value);

    // Destructor
    TransformerModel();
    TransformerModel(size_t vocab_size, size_t d_model, size_t n_layers, 
                   size_t n_heads, size_t d_ff, float dropout);
    ~TransformerModel();

    inline size_t vocab_size() const {
        return vocab_size_;
    }

private:
    // Private constructor
    // Implementation details
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
    
    // Model parameters
    size_t vocab_size_;
    size_t d_model_;
    size_t n_layers_;
    size_t n_heads_;
    size_t d_ff_;
    float dropout_;
};

} // namespace lm
