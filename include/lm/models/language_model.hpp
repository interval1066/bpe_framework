// include/lm/models/language_model.hpp
#pragma once

#include <vector>
#include <cstdint>
#include <string>
#include "../core/tensor.hpp"

namespace lm {

using TokenID = uint32_t;

class LanguageModel {
public:
    virtual ~LanguageModel() = default;
    
    // Pure virtual methods that must be implemented
    virtual std::vector<Tensor> get_parameters() const = 0;
    virtual void set_parameters(const std::vector<Tensor>& params) = 0;
    virtual Tensor forward(const std::vector<TokenID>& input) = 0;
    virtual Tensor forward(const std::vector<TokenID>& input, 
                          const std::vector<TokenID>& targets) = 0;
    
    // Optional virtual methods with default implementations
    virtual size_t get_vocab_size() const { return 0; }
    virtual size_t get_max_sequence_length() const { return 0; }
    
    // Serialization
    virtual void save(const std::string& path) const = 0;
    virtual void load(const std::string& path) = 0;
};

} // namespace lm

