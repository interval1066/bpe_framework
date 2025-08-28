#include "lm/models/language_model.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace lm {

LanguageModel::LanguageModel(size_t vocab_size, size_t embed_dim, 
                           size_t hidden_dim, size_t num_layers) {
    // Initialize model parameters
    parameters_["embedding.weight"] = Tensor::xavier({vocab_size, embed_dim}, true);
    
    // Initialize transformer layers
    for (size_t i = 0; i < num_layers; ++i) {
        // Initialize attention layers
        parameters_["transformer.layers." + std::to_string(i) + ".attention.query.weight"] = 
            Tensor::xavier({embed_dim, hidden_dim}, true);
        parameters_["transformer.layers." + std::to_string(i) + ".attention.query.bias"] = 
            Tensor::zeros({hidden_dim}, true);
            
        // Similarly initialize key, value, and output layers
        parameters_["transformer.layers." + std::to_string(i) + ".attention.key.weight"] = 
            Tensor::xavier({embed_dim, hidden_dim}, true);
        parameters_["transformer.layers." + std::to_string(i) + ".attention.value.weight"] = 
            Tensor::xavier({embed_dim, hidden_dim}, true);
        parameters_["transformer.layers." + std::to_string(i) + ".attention.output.weight"] = 
            Tensor::xavier({hidden_dim, embed_dim}, true);
            
        // Initialize feedforward layers
        parameters_["transformer.layers." + std::to_string(i) + ".feedforward.linear1.weight"] = 
            Tensor::xavier({embed_dim, hidden_dim * 4}, true);
        parameters_["transformer.layers." + std::to_string(i) + ".feedforward.linear2.weight"] = 
            Tensor::xavier({hidden_dim * 4, embed_dim}, true);
    }
    
    // Initialize output layer
    parameters_["output.weight"] = Tensor::xavier({embed_dim, vocab_size}, true);
    parameters_["output.bias"] = Tensor::zeros({vocab_size}, true);
    
    // Initialize transformer layers
    transformer_layers_.resize(num_layers);
}

std::vector<Tensor> LanguageModel::parameters() const {
    std::vector<Tensor> params;
    for (const auto& [name, tensor] : parameters_) {
        params.push_back(tensor);
    }
    return params;
}

std::unordered_map<std::string, Tensor> LanguageModel::named_parameters() const {
    return parameters_;
}

void LanguageModel::set_parameter(const std::string& name, const Tensor& param) {
    auto it = parameters_.find(name);
    if (it != parameters_.end()) {
        it->second = param;
    } else {
        throw std::runtime_error("Unknown parameter: " + name);
    }
}

void LanguageModel::save(const std::string& path) const {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file for writing: " + path);
    }
    
    // Write header
    const char magic[] = "LMOD";
    file.write(magic, 4);
    
    uint32_t version = 1;
    file.write(reinterpret_cast<const char*>(&version), sizeof(version));
    
    // Get named parameters
    auto params = named_parameters();
    uint32_t num_params = static_cast<uint32_t>(params.size());
    file.write(reinterpret_cast<const char*>(&num_params), sizeof(num_params));
    
    // Write each parameter
    for (const auto& [name, tensor] : params) {
        Tensor::write_string(file, name);
        tensor.serialize(file);
    }
}

void LanguageModel::load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file for reading: " + path);
    }
    
    // Read and verify header
    char magic[4];
    file.read(magic, 4);
    if (std::string(magic, 4) != "LMOD") {
        throw std::runtime_error("Invalid model file format");
    }
    
    uint32_t version;
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (version != 1) {
        throw std::runtime_error("Unsupported model version: " + std::to_string(version));
    }
    
    // Read number of parameters
    uint32_t num_params;
    file.read(reinterpret_cast<char*>(&num_params), sizeof(num_params));
    
    // Read each parameter
    for (uint32_t i = 0; i < num_params; ++i) {
        std::string name = Tensor::read_string(file);
        Tensor tensor;
        tensor.deserialize(file);
        
        // Set the parameter
        set_parameter(name, tensor);
    }
}

void LanguageModel::train() {
    is_training_ = true;
}

void LanguageModel::eval() {
    is_training_ = false;
}

Tensor LanguageModel::forward(const Tensor& input) {
    // This is a simplified forward pass implementation
    // Embedding layer
    Tensor embedded = embedding_forward(input);
    
    // Transformer layers
    Tensor transformer_out = embedded;
    for (auto& layer : transformer_layers_) {
        transformer_out = layer.forward(transformer_out);
    }
    
    // Output layer
    Tensor output = output_forward(transformer_out);
    
    return output;
}

Tensor LanguageModel::embedding_forward(const Tensor& input) {
    // Simplified embedding forward pass
    auto& embedding_weight = parameters_.at("embedding.weight");
    
    // Create shape vector explicitly to avoid ambiguous constructor call
    std::vector<size_t> shape = {
        static_cast<size_t>(input.size()), 
        static_cast<size_t>(embedding_weight.shape()[1])
    };
    Tensor result(shape);
    
    for (int i = 0; i < input.size(); ++i) {
        int token_id = static_cast<int>(input(i));
        if (token_id >= 0 && token_id < embedding_weight.shape()[0]) {
            result.data().row(i) = embedding_weight.data().row(token_id);
        }
    }
    
    return result;
}

Tensor LanguageModel::output_forward(const Tensor& input) {
    // Simplified output forward pass
    auto& output_weight = parameters_.at("output.weight");
    auto& output_bias = parameters_.at("output.bias");
    
    Tensor result = input.matmul(output_weight) + output_bias;
    return result;
}

} // namespace lm

