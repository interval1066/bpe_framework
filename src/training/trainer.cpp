#include "lm/training/trainer.hpp"
#include "lm/generation/sampler.hpp"
#include "lm/conversation.hpp"
#include <queue>
#include <iostream>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <functional>

namespace lm {

LanguageModelTrainer::LanguageModelTrainer(const BPETokenizer& tokenizer,
                                         size_t embedding_dim,
                                         size_t hidden_dim,
                                         size_t num_layers)
    : tokenizer_(tokenizer),
      model_(tokenizer.vocab_size(), embedding_dim, hidden_dim, num_layers),
      optimizer_(0.001, 0.9, 0.999, 1e-8) {
    // Constructor implementation
    std::cout << "LanguageModelTrainer initialized with:" << std::endl;
    std::cout << "  Embedding dim: " << embedding_dim << std::endl;
    std::cout << "  Hidden dim: " << hidden_dim << std::endl;
    std::cout << "  Num layers: " << num_layers << std::endl;
    std::cout << "  Vocab size: " << tokenizer.vocab_size() << std::endl;
}

std::pair<std::vector<std::string>, std::vector<std::string>> 
LanguageModelTrainer::split_data(const std::vector<std::string>& corpus, float validation_split) {
    if (validation_split <= 0.0f || validation_split >= 1.0f) {
        throw std::invalid_argument("Validation split must be between 0 and 1");
    }
    
    size_t validation_size = static_cast<size_t>(corpus.size() * validation_split);
    size_t train_size = corpus.size() - validation_size;
    
    // Create indices and shuffle
    std::vector<size_t> indices(corpus.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        indices[i] = i;
    }
    
    std::shuffle(indices.begin(), indices.end(), std::default_random_engine(42));
    
    // Split the data
    std::vector<std::string> train_data;
    std::vector<std::string> val_data;
    
    train_data.reserve(train_size);
    val_data.reserve(validation_size);
    
    for (size_t i = 0; i < train_size; ++i) {
        train_data.push_back(corpus[indices[i]]);
    }
    
    for (size_t i = train_size; i < corpus.size(); ++i) {
        val_data.push_back(corpus[indices[i]]);
    }
    
    return {train_data, val_data};
}

float LanguageModelTrainer::run_validation(const std::vector<std::string>& validation_data,
                                         size_t batch_size,
                                         size_t sequence_length) {
    if (validation_data.empty()) {
        return 0.0f;
    }
    
    model_.eval();  // Set model to evaluation mode
    float total_val_loss = 0.0f;
    size_t num_val_batches = 0;
    
    for (size_t i = 0; i < validation_data.size(); i += batch_size) {
        size_t end = std::min(i + batch_size, validation_data.size());
        std::vector<std::string> batch_texts(validation_data.begin() + i, 
                                           validation_data.begin() + end);
        
        Tensor batch = prepare_batch(batch_texts, sequence_length);
        Tensor input = batch.slice(0, sequence_length - 1, 0);
        Tensor target = batch.slice(1, sequence_length - 1, 0);
        
        // Forward pass without gradient computation
        Tensor logits = model_.forward(input);
        float loss = compute_loss(logits, target);
        total_val_loss += loss;
        num_val_batches++;
    }
    
    model_.train();  // Set model back to training mode
    return total_val_loss / num_val_batches;
}

void LanguageModelTrainer::train(const std::vector<std::string>& corpus, 
                               size_t epochs, 
                               size_t batch_size, 
                               size_t sequence_length,
                               float validation_split,
                               size_t validation_freq,
                               size_t early_stopping_patience,
                               const ProgressCallback& callback) {
    
    // Split data into training and validation sets
    auto [train_data, val_data] = split_data(corpus, validation_split);
    
    std::cout << "Training on " << train_data.size() << " samples, "
              << "validating on " << val_data.size() << " samples" << std::endl;
    
    model_.train();
    train_loss_history_.clear();
    val_loss_history_.clear();
    
    size_t best_epoch = 0;
    float best_val_loss = std::numeric_limits<float>::max();
    size_t epochs_without_improvement = 0;
    
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0.0;
        size_t num_batches = 0;
        
        // Shuffle training corpus
        std::vector<std::string> shuffled_corpus = train_data;
        std::shuffle(shuffled_corpus.begin(), shuffled_corpus.end(), 
                    std::default_random_engine(42 + epoch));  // Different seed each epoch
        
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
            
            // Update parameters
            auto params = model_.parameters();
            optimizer_.step(params);
            optimizer_.zero_grad(params);
            
            num_batches++;
            
            if (num_batches % 100 == 0) {
                std::cout << "Epoch " << epoch + 1 << ", Batch " << num_batches 
                          << ", Loss: " << loss << std::endl;
            }
        }
        
        float avg_train_loss = total_loss / num_batches;
        train_loss_history_.push_back(avg_train_loss);
        
        // Run validation if needed
        float avg_val_loss = 0.0f;
        if (val_data.size() > 0 && (epoch % validation_freq == 0 || epoch == epochs - 1)) {
            avg_val_loss = run_validation(val_data, batch_size, sequence_length);
            val_loss_history_.push_back(avg_val_loss);
            
            // Check for improvement for early stopping
            if (avg_val_loss < best_val_loss) {
                best_val_loss = avg_val_loss;
                best_epoch = epoch;
                epochs_without_improvement = 0;
            } else {
                epochs_without_improvement++;
            }
        }
        
        std::cout << "Epoch " << epoch + 1 << " completed. "
                  << "Train Loss: " << avg_train_loss 
                  << ", Val Loss: " << avg_val_loss << std::endl;
        
        // Call progress callback if provided
        if (callback) {
            callback(epoch + 1, avg_train_loss, avg_val_loss, optimizer_.get_learning_rate());
        }
        
        // Check for early stopping
        if (early_stopping_patience > 0 && 
            epochs_without_improvement >= early_stopping_patience) {
            std::cout << "Early stopping triggered at epoch " << epoch + 1 
                      << ". Best validation loss was " << best_val_loss 
                      << " at epoch " << best_epoch + 1 << std::endl;
            break;
        }
    }
}

float LanguageModelTrainer::evaluate(const std::vector<std::string>& corpus,
                                   size_t batch_size,
                                   size_t sequence_length) {
    return run_validation(corpus, batch_size, sequence_length);
}

std::pair<float, float> LanguageModelTrainer::test(const std::vector<std::string>& corpus,
                                                 size_t batch_size,
                                                 size_t sequence_length) {
    float loss = evaluate(corpus, batch_size, sequence_length);
    
    // For language models, we often report perplexity as well
    float perplexity = std::exp(loss);
    
    return {loss, perplexity};
}

void LanguageModelTrainer::save_model(const std::string& path) {
    model_.save(path);
    std::cout << "Model saved to: " << path << std::endl;
}

void LanguageModelTrainer::load_model(const std::string& path) {
    model_.load(path);
    std::cout << "Model loaded from: " << path << std::endl;
}

Tensor LanguageModelTrainer::prepare_batch(const std::vector<std::string>& texts, 
                                         size_t sequence_length) {
    std::vector<std::vector<TokenID>> tokenized_texts;
    
    // Tokenize all texts
    for (const auto& text : texts) {
        tokenized_texts.push_back(tokenizer_.encode(text));
    }
    
    // Create batch tensor
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

Tensor LanguageModelTrainer::prepare_inference_batch(const std::vector<std::vector<int>>& token_sequences, 
                                                   size_t sequence_length) {
    size_t batch_size = token_sequences.size();
    
    // Create batch tensor
    std::vector<size_t> shape = {sequence_length, batch_size};
    Tensor batch(shape);
    
    // Fill batch
    for (size_t i = 0; i < batch_size; ++i) {
        const auto& tokens = token_sequences[i];
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

std::vector<int> LanguageModelTrainer::generate_tokens(const std::string& prompt, 
                                                     size_t max_length, 
                                                     Sampler& sampler,
                                                     size_t sequence_length) {
    model_.eval();
    
    // Tokenize prompt
    std::vector<TokenID> temp_tokens = tokenizer_.encode(prompt);
    std::vector<int> tokens(temp_tokens.begin(), temp_tokens.end());
    std::vector<int> generated = tokens;
    
    // Get EOS token ID
    int eos_token_id = static_cast<int>(tokenizer_.eos_token_id());
    
    for (size_t i = 0; i < max_length; ++i) {
        // Prepare input (last sequence_length tokens)
        std::vector<int> input_tokens;
        if (generated.size() > sequence_length) {
            input_tokens = std::vector<int>(generated.end() - sequence_length, generated.end());
        } else {
            input_tokens = generated;
        }
        
        // Create batch
        std::vector<std::vector<int>> batch = {input_tokens};
        Tensor input = prepare_inference_batch(batch, sequence_length);
        
        // Forward pass
        Tensor logits = model_.forward(input);
        
        // Get last token logits
        size_t last_pos = input_tokens.size() - 1;
        Tensor last_logits_slice = logits.slice(last_pos, last_pos + 1, 0);
        
        // Extract the values into a 1D tensor
        size_t vocab_size = last_logits_slice.shape().back();
        Tensor last_logits({vocab_size});
        for (size_t k = 0; k < vocab_size; k++) {
            last_logits(k) = last_logits_slice(0, 0, k);
        }
        
        // Sample next token
        int next_token = sampler.sample(last_logits);
        generated.push_back(next_token);
        
        // Stop if we generate an end-of-sequence token
        if (eos_token_id != 0 && next_token == eos_token_id) {
            break;
        }
    }
    
    model_.train();
    return generated;
}

std::string LanguageModelTrainer::generate(const std::string& prompt, 
                                         size_t max_length, 
                                         Sampler& sampler,
                                         size_t sequence_length) {
    std::vector<int> tokens = generate_tokens(prompt, max_length, sampler, sequence_length);
    
    // Convert int tokens to TokenID for decoding
    std::vector<TokenID> decode_tokens(tokens.begin(), tokens.end());
    return tokenizer_.decode(decode_tokens);
}

std::vector<std::string> LanguageModelTrainer::generate_batch(const std::vector<std::string>& prompts, 
                                                            size_t max_length, 
                                                            Sampler& sampler,
                                                            size_t sequence_length,
                                                            size_t batch_size) {
    model_.eval();
    
    std::vector<std::string> results;
    std::vector<std::vector<int>> all_generated;
    
    // Get EOS token ID
    int eos_token_id = static_cast<int>(tokenizer_.eos_token_id());
    
    // Initialize with prompts
    for (const auto& prompt : prompts) {
        std::vector<TokenID> temp_tokens = tokenizer_.encode(prompt);
        std::vector<int> tokens(temp_tokens.begin(), temp_tokens.end());
        all_generated.push_back(tokens);
    }
    
    // Create a mask for completed sequences
    std::vector<bool> completed(prompts.size(), false);
    
    for (size_t step = 0; step < max_length; ++step) {
        // Check if all sequences are completed
        if (std::all_of(completed.begin(), completed.end(), [](bool v) { return v; })) {
            break;
        }
        
        // Process in batches
        for (size_t i = 0; i < prompts.size(); i += batch_size) {
            size_t end = std::min(i + batch_size, prompts.size());
            
            // Prepare batch
            std::vector<std::vector<int>> batch_inputs;
            std::vector<size_t> batch_indices;
            
            for (size_t j = i; j < end; ++j) {
                if (completed[j]) continue;
                
                // Get last sequence_length tokens
                std::vector<int> input_tokens;
                if (all_generated[j].size() > sequence_length) {
                    input_tokens = std::vector<int>(all_generated[j].end() - sequence_length, all_generated[j].end());
                } else {
                    input_tokens = all_generated[j];
                }
                
                batch_inputs.push_back(input_tokens);
                batch_indices.push_back(j);
            }
            
            if (batch_inputs.empty()) continue;
            
            // Create tensor batch
            Tensor input = prepare_inference_batch(batch_inputs, sequence_length);
            
            // Forward pass
            Tensor logits = model_.forward(input);
            
            // Get last token logits for each sequence in batch
            for (size_t b = 0; b < batch_inputs.size(); ++b) {
                size_t last_pos = batch_inputs[b].size() - 1;
                Tensor last_logits_slice = logits.slice(last_pos, last_pos + 1, b);
                
                // Extract the values into a 1D tensor
                size_t vocab_size = last_logits_slice.shape().back();
                Tensor last_logits({vocab_size});
                for (size_t k = 0; k < vocab_size; k++) {
                    last_logits(k) = last_logits_slice(0, 0, k);
                }
                
                // Sample next token
                int next_token = sampler.sample(last_logits);
                size_t orig_idx = batch_indices[b];
                all_generated[orig_idx].push_back(next_token);
                
                // Mark as completed if EOS token is generated
                if (eos_token_id != 0 && next_token == eos_token_id) {
                    completed[orig_idx] = true;
                }
            }
        }
    }
    
    // Decode all generated sequences
    for (const auto& tokens : all_generated) {
        // Convert int tokens to TokenID for decoding
        std::vector<TokenID> decode_tokens(tokens.begin(), tokens.end());
        results.push_back(tokenizer_.decode(decode_tokens));
    }
    
    model_.train();
    return results;
}

std::vector<std::string> LanguageModelTrainer::prepare_conversation_corpus(
    const std::vector<Conversation>& conversations,
    size_t context_turns,
    size_t max_sequence_length) {
    
    std::vector<std::string> corpus;
    
    for (const auto& conv : conversations) {
        if (conv.turns.size() < 2) continue;
        
        // Create training examples from conversation turns
        for (size_t i = 1; i < conv.turns.size(); i++) {
            // Get context window
            size_t start_idx = (i > context_turns) ? i - context_turns : 0;
            auto context_window = std::vector<ConversationTurn>(
                conv.turns.begin() + start_idx, conv.turns.begin() + i);
            
            // Format context and target
            std::string context = conversation_utils::extract_text(context_window);
            std::string target = conv.turns[i].text;
            
            // Create training example
            std::string example = context + " " + target;
            
            // Truncate if too long (optional)
            if (max_sequence_length > 0 && example.size() > max_sequence_length) {
                example = example.substr(0, max_sequence_length);
            }
            
            corpus.push_back(example);
        }
    }
    
    return corpus;
}

void LanguageModelTrainer::train_on_conversations(
    const std::vector<Conversation>& conversations,
    size_t epochs,
    size_t batch_size,
    size_t sequence_length,
    size_t context_turns,
    float validation_split,
    size_t validation_freq,
    size_t early_stopping_patience,
    const ProgressCallback& callback) {
    
    // Prepare training corpus from conversations
    std::vector<std::string> corpus = prepare_conversation_corpus(
        conversations, context_turns, sequence_length);
    
    // Use existing training method
    train(corpus, epochs, batch_size, sequence_length, 
          validation_split, validation_freq, early_stopping_patience, callback);
}

std::string LanguageModelTrainer::continue_conversation(
    const Conversation& conversation,
    Sampler& sampler,
    size_t max_length,
    size_t sequence_length,
    size_t context_turns) {
    
    // Get the most recent turns as context
    auto context_window = conversation_utils::get_context_window(
        conversation.turns, context_turns);
    
    // Format context
    std::string context = conversation_utils::extract_text(context_window);
    
    // Generate continuation
    return generate(context, max_length, sampler, sequence_length);
}

} // namespace lm
