// lm/models/language_model.cpp
#include "lm/models/language_model.hpp"
#include "lm/optimizers/adam.hpp"
#include <random>

namespace lm {

LanguageModel::LanguageModel(size_t vocab_size, size_t embedding_dim, 
                           size_t hidden_dim, size_t num_layers)
    : vocab_size_(vocab_size), embedding_dim_(embedding_dim),
      hidden_dim_(hidden_dim), num_layers_(num_layers), is_training_(true) {
    
    // Initialize embedding layer
    embedding_weight_ = Tensor::xavier({vocab_size, embedding_dim}, true);
    
    // Initialize LSTM layers
    size_t gate_size = 4 * hidden_dim;
    lstm_weight_ih_ = Tensor::xavier({gate_size, embedding_dim}, true);
    lstm_weight_hh_ = Tensor::xavier({gate_size, hidden_dim}, true);
    lstm_bias_ih_ = Tensor::zeros({gate_size}, true);
    lstm_bias_hh_ = Tensor::zeros({gate_size}, true);
    
    // Initialize output layer
    output_weight_ = Tensor::xavier({vocab_size, hidden_dim}, true);
    output_bias_ = Tensor::zeros({vocab_size}, true);
}

Tensor LanguageModel::forward(const Tensor& input) {
    // Input shape: [sequence_length, batch_size]
    // Get sequence length and batch size
    size_t seq_len = input.shape()[0];
    size_t batch_size = input.shape()[1];
    
    // Embedding layer
    Tensor embedded = embedding_weight_.index_select(input);  // [seq_len, batch_size, embedding_dim]
    
    // LSTM layer (simplified implementation)
    Tensor hidden = Tensor::zeros({num_layers_, batch_size, hidden_dim});
    Tensor cell = Tensor::zeros({num_layers_, batch_size, hidden_dim});
    
    Tensor output;
    for (size_t t = 0; t < seq_len; ++t) {
        // Get current time step
        Tensor x_t = embedded.slice(t, 1, 0);  // [batch_size, embedding_dim]
        
        // LSTM computation (simplified)
        for (size_t layer = 0; layer < num_layers_; ++layer) {
            Tensor h_prev = hidden.slice(layer, 1, 0);
            Tensor c_prev = cell.slice(layer, 1, 0);
            
            // Gates computation
            Tensor gates = x_t.matmul(lstm_weight_ih_.transpose()) + 
                          h_prev.matmul(lstm_weight_hh_.transpose()) +
                          lstm_bias_ih_ + lstm_bias_hh_;
            
            // Split gates
            Tensor i = gates.slice(0, hidden_dim, 1).sigmoid();
            Tensor f = gates.slice(hidden_dim, hidden_dim, 1).sigmoid();
            Tensor g = gates.slice(2 * hidden_dim, hidden_dim, 1).tanh();
            Tensor o = gates.slice(3 * hidden_dim, hidden_dim, 1).sigmoid();
            
            // Update cell state
            Tensor c_next = f * c_prev + i * g;
            
            // Update hidden state
            Tensor h_next = o * c_next.tanh();
            
            // Store states
            hidden.slice(layer, 1, 0) = h_next;
            cell.slice(layer, 1, 0) = c_next;
            
            x_t = h_next;  // Output of this layer is input to next layer
        }
        
        // Store output for this time step
        if (t == 0) {
            output = x_t.unsqueeze(0);  // Add sequence dimension
        } else {
            output = output.concatenate(x_t.unsqueeze(0), 0);
        }
    }
    
    // Output layer
    Tensor logits = output.matmul(output_weight_.transpose()) + output_bias_;
    return logits;
}

std::vector<Tensor> LanguageModel::parameters() const {
    return {
        embedding_weight_,
        lstm_weight_ih_,
        lstm_weight_hh_,
        lstm_bias_ih_,
        lstm_bias_hh_,
        output_weight_,
        output_bias_
    };
}

void LanguageModel::train() {
    is_training_ = true;
}

void LanguageModel::eval() {
    is_training_ = false;
}

} // namespace lm
