// lm/models/language_model.hpp
#pragma once

#include "../core/tensor.hpp"
#include "../tokenizer/bpe_tokenizer.hpp"
#include <vector>

namespace lm {

class LanguageModel {
public:
    LanguageModel(size_t vocab_size, size_t embedding_dim, size_t hidden_dim, size_t num_layers);
    
    Tensor forward(const Tensor& input);
    Tensor operator()(const Tensor& input) { return forward(input); }

    void save(const std::string& path) const;
    void load(const std::string& path);
    
    // Parameter access methods
    std::vector<Tensor> parameters() const;
    std::unordered_map<std::string, Tensor> named_parameters() const;
    void set_parameter(const std::string& name, const Tensor& param);
    
    void train();
    void eval();
    
private:
    size_t vocab_size_, embedding_dim_, hidden_dim_, num_layers_;
    
    // Model parameters
    Tensor embedding_weight_;
    Tensor lstm_weight_ih_;
    Tensor lstm_weight_hh_;
    Tensor lstm_bias_ih_;
    Tensor lstm_bias_hh_;
    Tensor output_weight_;
    Tensor output_bias_;
    
    bool is_training_;
    std::unordered_map<std::string, Tensor> parameters_;

};

} // namespace lm
