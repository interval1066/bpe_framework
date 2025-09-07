#pragma once

#include "lm/core/tensor.hpp"
#include <string>
#include <unordered_map>
#include <vector>

namespace lm {

class LanguageModel {
public:
    LanguageModel(size_t vocab_size, size_t embed_dim, size_t hidden_dim, size_t num_layers);
    
    Tensor forward(const Tensor& input);
    void train();
    void eval();
    
    // Parameter access methods
    std::vector<Tensor> parameters() const;
    std::unordered_map<std::string, Tensor> named_parameters() const;
    void set_parameter(const std::string& name, const Tensor& param);

    virtual std::vector<Tensor> get_parameters() const = 0;
    virtual void set_parameters(const std::vector<Tensor>& params) = 0;

    // Serialization methods
    void save(const std::string& path) const;
    void load(const std::string& path);

private:
    std::unordered_map<std::string, Tensor> parameters_;
    bool is_training_ = true;
    
    // Placeholder for actual model components
    Tensor embedding_forward(const Tensor& input);
    Tensor output_forward(const Tensor& input);
    
    // Placeholder for transformer layers
    struct TransformerLayer {
        Tensor forward(const Tensor& input) { return input; } // Simplified
    };
    std::vector<TransformerLayer> transformer_layers_;
};

} // namespace lm
