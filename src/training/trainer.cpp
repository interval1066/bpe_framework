#include "lm/training/trainer.hpp"
#include <iostream>
#include <random>
#include <algorithm>

namespace lm {

LanguageModelTrainer::LanguageModelTrainer(const BPETokenizer& tokenizer,
                                         size_t embedding_dim,
                                         size_t hidden_dim,
                                         size_t num_layers)
    : tokenizer_(tokenizer),  // Store reference
      model_(tokenizer.vocab_size(), embedding_dim, hidden_dim, num_layers),
      optimizer_(0.001, 0.9, 0.999, 1e-8) {}

void LanguageModelTrainer::train(const std::vector<std::string>& corpus, 
                               size_t epochs, 
                               size_t batch_size, 
                               size_t sequence_length) {
    
    model_.train();
    
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0.0;
        size_t num_batches = 0;
        
        // Shuffle corpus
        std::vector<std::string> shuffled_corpus = corpus;
        std::shuffle(shuffled_corpus.begin(), shuffled_corpus.end(), 
                    std::default_random_engine(42));
        
        // Process in batches
        for (size_t i = 0; i < shuffled_corpus.size(); i += batch_size) {
            size_t end = std::min(i + batch_size, shuffled_corpus.size());
            std::vector<std::string> batch_texts(shuffled_corpus.begin() + i, 
                                               shuffled_corpus.begin() + end);
            
            // Prepare batch
            Tensor batch = prepare_batch(batch_texts, sequence_length);
            
            // Split into input and target
            Tensor input = batch.slice(0, sequence_length - 1, 0);
            Tensor target = batch.slice(1, sequence_length - 1, 0);
            
            // Forward pass
            Tensor logits = model_.forward(input);
            
            // Compute loss
            float loss = compute_loss(logits, target);
            total_loss += loss;
            
            // Backward pass
            logits.backward();
            
            // Update parameters - store in variable to avoid rvalue reference issue
            auto params = model_.parameters();
            optimizer_.step(params);
            optimizer_.zero_grad(params);
            
            num_batches++;
            
            if (num_batches % 100 == 0) {
                std::cout << "Epoch " << epoch + 1 << ", Batch " << num_batches 
                          << ", Loss: " << loss << std::endl;
            }
        }
        
        std::cout << "Epoch " << epoch + 1 << " completed. Average loss: " 
                  << total_loss / num_batches << std::endl;
    }
}

Tensor LanguageModelTrainer::prepare_batch(const std::vector<std::string>& texts, 
                                         size_t sequence_length) {
    std::vector<std::vector<TokenID>> tokenized_texts;
    
    // Tokenize all texts
    for (const auto& text : texts) {
        tokenized_texts.push_back(tokenizer_.encode(text));
    }
    
    // Create batch tensor - fix ambiguous constructor
    std::vector<size_t> shape = {sequence_length, texts.size()};
    Tensor batch(shape);
    
    // Fill batch
    for (size_t i = 0; i < texts.size(); ++i) {
        const auto& tokens = tokenized_texts[i];
        for (size_t j = 0; j < sequence_length; ++j) {
            if (j < tokens.size()) {
                batch(j, i) = static_cast<float>(tokens[j]);
            } else {
                // Padding
                batch(j, i) = 0.0f;
            }
        }
    }
    
    return batch;
}

float LanguageModelTrainer::compute_loss(const Tensor& logits, const Tensor& targets) {
    // Cross-entropy loss
    Tensor log_probs = logits.softmax(-1);
    
    // Gather the log probabilities of the target classes
    Tensor loss = Tensor::zeros({1});
    size_t batch_size = targets.shape()[1];
    size_t seq_length = targets.shape()[0];
    
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < seq_length; ++j) {
            int target_class = static_cast<int>(targets(j, i));
            if (target_class != 0) {  // Skip padding
                loss(0) -= log_probs(j, i, target_class);
            }
        }
    }
    
    // Average loss
    return loss(0) / (batch_size * seq_length);
}

void LanguageModelTrainer::save_model(const std::string& path) {
    // Implementation for saving model parameters
}

void LanguageModelTrainer::load_model(const std::string& path) {
    // Implementation for loading model parameters
}

} // namespace lm
