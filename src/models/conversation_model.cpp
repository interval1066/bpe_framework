#include "lm/models/conversation_model.hpp"
#include "lm/data/training_data.hpp"
#include "lm/context_manager.hpp"
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <cereal/archives/binary.hpp>
#include <cereal/types/string.hpp>
#include <limits>
#include <algorithm>
#include <cmath>

namespace lm {

// Helper function to get current timestamp
std::string ConversationModel::get_current_timestamp() const {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %X");
    return ss.str();
}

// Improved learning rate scheduler
float ConversationModel::calculate_learning_rate(size_t epoch, size_t total_epochs, float base_lr) const {
    // Warmup for first 10% of training
    if (epoch < total_epochs * 0.1f) {
        return base_lr * (float(epoch) / (total_epochs * 0.1f));
    }
    
    // Cosine decay after warmup
    float progress = float(epoch - total_epochs * 0.1f) / (total_epochs * 0.9f);
    return base_lr * 0.5f * (1.0f + std::cos(progress * 3.14159265359f));
}

// Validate training example
bool ConversationModel::validate_training_example(const std::vector<TokenID>& input, 
                                                 const std::vector<TokenID>& target) const {
    if (input.empty() || target.empty()) {
        return false;
    }
    
    if (input.size() > 100 || target.size() > 100) {
        return false;
    }
    
    // Check for invalid token IDs
    for (TokenID id : input) {
        if (id >= transformer_->vocab_size()) {
            return false;
        }
    }
    for (TokenID id : target) {
        if (id >= transformer_->vocab_size()) {
            return false;
        }
    }
    
    return true;
}

// Constructor
ConversationModel::ConversationModel(size_t vocab_size, 
                                   size_t d_model, 
                                   size_t n_layers, 
                                   size_t n_heads,
                                   size_t d_ff,
                                   float dropout)
    : transformer_(std::make_unique<TransformerModel>(vocab_size, d_model, n_layers, n_heads, d_ff, dropout)),
      pad_token_id_(1) {
    // Initialize with conservative defaults
    max_grad_norm_ = 1.0f;
    lr_decay_ = 0.95f;
    weight_decay_ = 0.01f;
}

void ConversationModel::train(const TrainingDataset& dataset, 
                             size_t num_epochs, 
                             float learning_rate,
                             const std::string& resume_checkpoint) {
    // CRITICAL: Use much smaller learning rate
    learning_rate = 0.0001f; // 1e-4
    
    if (!tokenizer_) {
        throw std::runtime_error("Tokenizer not set before training");
    }
    
    training_timestamp_ = get_current_timestamp();
    const auto& examples = dataset.examples();
    size_t total_examples = examples.size();
    
    if (verbose_) {
        std::cout << "[" << training_timestamp_ << "] Starting training with " 
                  << total_examples << " examples for " << num_epochs << " epochs" << std::endl;
        std::cout << "Using stable learning rate: " << learning_rate << std::endl;
    }
    
    if (total_examples == 0) {
        std::cerr << "Warning: No training examples provided!" << std::endl;
        return;
    }
    
    float best_loss = std::numeric_limits<float>::max();
    size_t examples_skipped = 0;
    size_t examples_processed = 0;
    
    // Use only same-length examples for stability
    lm::TrainingDataset stable_dataset;
    for (const auto& example : examples) {
        auto input_tokens = tokenizer_->encode(example.input);
        auto target_tokens = tokenizer_->encode(example.target);
        
        // Only use examples where input and target have same length
        if (input_tokens.size() == target_tokens.size() && 
            !input_tokens.empty() && 
            input_tokens.size() <= 20) { // Limit sequence length
            stable_dataset.add_example(example.input, example.target);
        }
    }
    
    if (stable_dataset.size() == 0) {
        std::cerr << "No valid same-length training examples found!" << std::endl;
        return;
    }
    
    if (verbose_) {
        std::cout << "Using " << stable_dataset.size() << " same-length examples for stable training" << std::endl;
    }
    
    const auto& stable_examples = stable_dataset.examples();
    
    // Training loop
    for (size_t epoch = 0; epoch < num_epochs; epoch++) {
        float epoch_loss = 0.0f;
        examples_processed = 0;
        
        if (verbose_) {
            std::cout << "[" << get_current_timestamp() << "] Epoch " << (epoch + 1) 
                      << "/" << num_epochs << std::endl;
        }
        
        for (const auto& example : stable_examples) {
            try {
                // Single training step
                float loss = transformer_->train_step(
                    tokenizer_->encode(example.input),
                    tokenizer_->encode(example.target),
                    learning_rate
                );
                
                // Check for exploding loss
                if (std::isnan(loss) || std::isinf(loss) || loss > 1e5f) {
                    examples_skipped++;
                    if (verbose_) {
                        std::cout << "Skipping example due to high loss: " << loss << std::endl;
                    }
                    continue;
                }
                
                epoch_loss += loss;
                examples_processed++;
                
                if (verbose_ && examples_processed % 10 == 0) {
                    std::cout << "Processed " << examples_processed << " examples, loss: " << loss << std::endl;
                }
                
            } catch (const std::exception& e) {
                examples_skipped++;
                if (verbose_) {
                    std::cerr << "Error training example: " << e.what() << std::endl;
                }
            }
        }
        
        // Calculate average epoch loss
        if (examples_processed > 0) {
            epoch_loss /= examples_processed;
            
            if (verbose_) {
                std::cout << "Epoch " << (epoch + 1) << " average loss: " << epoch_loss 
                          << " (skipped " << examples_skipped << " examples)" << std::endl;
            }
            
            // Update best loss
            if (epoch_loss < best_loss) {
                best_loss = epoch_loss;
                best_loss_ = best_loss;
                if (verbose_) {
                    std::cout << "New best loss: " << best_loss << std::endl;
                }
                
                // Save best model
                save_checkpoint("best_model.checkpoint");
            }
            
            // Stop if loss becomes unstable
            if (epoch_loss > 1e4f) {
                std::cerr << "Loss becoming unstable: " << epoch_loss << " - stopping early" << std::endl;
                break;
            }
        }
        
        training_epochs_++;
        examples_skipped = 0;
        
        // Gradually reduce learning rate
        learning_rate *= 0.9f;
        if (learning_rate < 1e-6f) learning_rate = 1e-6f;
    }
    
    // Final save
    save_checkpoint("final_model.checkpoint");
    
    if (verbose_) {
        std::cout << "[" << get_current_timestamp() << "] Training completed. Best loss: " 
                  << best_loss_ << std::endl;
        std::cout << "Total epochs: " << training_epochs_ << std::endl;
    }
}

// Generate response method
std::string ConversationModel::generate_response(const std::string& input) {
    if (!tokenizer_) {
        throw std::runtime_error("Tokenizer not set");
    }
    
    try {
        // Encode input
        auto input_tokens = tokenizer_->encode(input);
        
        if (input_tokens.empty()) {
            return "I'm not sure how to respond to that.";
        }
        
        // Generate response with temperature parameter
        float temperature = 0.8f; // Default temperature value
        auto response_tokens = transformer_->generate(input_tokens, max_response_length_, temperature);
        
        // Decode response
        std::string response = tokenizer_->decode(response_tokens);
        
        // Validate response
        if (response.empty() || response.length() < 2) {
            return "I'm not sure how to respond to that.";
        }
        
        return response;
    } catch (const std::exception& e) {
        std::cerr << "Error generating response: " << e.what() << std::endl;
        return "I'm experiencing technical difficulties. Please try again.";
    }
}

// IMPROVED Save checkpoint method
bool ConversationModel::save_checkpoint(const std::string& filename) {
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs.is_open()) {
        std::cerr << "Failed to open checkpoint file: " << filename << std::endl;
        return false;
    }
    
    try {
        cereal::BinaryOutputArchive archive(ofs);
        
        // Serialize the transformer model
        if (transformer_) {
            std::stringstream model_stream;
            transformer_->serialize(model_stream);
            std::string model_data = model_stream.str();
            archive(model_data);
        } else {
            std::string empty;
            archive(empty);
        }
        
        // Serialize training metadata and configuration
        archive(
            training_epochs_,
            best_loss_,
            training_timestamp_,
            max_response_length_,
            verbose_,
            max_grad_norm_,    // Save training configuration
            lr_decay_,
            weight_decay_
        );
        
        // Serialize tokenizer information if available
        if (tokenizer_) {
            size_t vocab_size = tokenizer_->vocab_size();
            archive(vocab_size);
            
            std::string temp_filename = "temp_tokenizer_checkpoint.bin";
            if (tokenizer_->save(temp_filename)) {
                std::ifstream temp_file(temp_filename, std::ios::binary);
                if (temp_file.is_open()) {
                    std::string tokenizer_data(
                        (std::istreambuf_iterator<char>(temp_file)),
                        std::istreambuf_iterator<char>()
                    );
                    temp_file.close();
                    archive(tokenizer_data);
                    std::remove(temp_filename.c_str());
                } else {
                    std::string empty;
                    archive(empty);
                }
            } else {
                std::string empty;
                archive(empty);
            }
        } else {
            size_t vocab_size = 0;
            std::string empty;
            archive(vocab_size, empty);
        }
        
        if (verbose_) {
            std::cout << "Checkpoint saved: " << filename << std::endl;
            std::cout << "  Epochs: " << training_epochs_ << std::endl;
            std::cout << "  Best loss: " << best_loss_ << std::endl;
            std::cout << "  Timestamp: " << training_timestamp_ << std::endl;
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error saving checkpoint: " << e.what() << std::endl;
        return false;
    }
}

// IMPROVED Load checkpoint method
bool ConversationModel::load_checkpoint(const std::string& filename) {
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs.is_open()) {
        std::cerr << "Failed to open checkpoint file: " << filename << std::endl;
        return false;
    }
    
    try {
        cereal::BinaryInputArchive archive(ifs);
        
        // Deserialize the transformer model
        std::string model_data;
        archive(model_data);
        
        if (!model_data.empty() && transformer_) {
            std::stringstream model_stream(model_data);
            transformer_->deserialize(model_stream);
        }
        
        // Deserialize training metadata and configuration
        archive(
            training_epochs_,
            best_loss_,
            training_timestamp_,
            max_response_length_,
            verbose_,
            max_grad_norm_,    // Load training configuration
            lr_decay_,
            weight_decay_
        );
        
        // Deserialize tokenizer information if available
        size_t saved_vocab_size;
        std::string tokenizer_data;
        archive(saved_vocab_size, tokenizer_data);
        
        if (tokenizer_ && !tokenizer_data.empty()) {
            // Verify vocabulary size matches
            if (saved_vocab_size != tokenizer_->vocab_size()) {
                std::cerr << "Warning: Vocabulary size mismatch. Saved: " 
                          << saved_vocab_size << ", Current: " << tokenizer_->vocab_size() 
                          << std::endl;
            }
            
            // Write tokenizer data to a temporary file and load it
            std::string temp_filename = "temp_tokenizer_checkpoint.bin";
            std::ofstream temp_file(temp_filename, std::ios::binary);
            if (temp_file.is_open()) {
                temp_file.write(tokenizer_data.data(), tokenizer_data.size());
                temp_file.close();
                tokenizer_->load(temp_filename);
                std::remove(temp_filename.c_str());
            } else {
                std::cerr << "Warning: Failed to create temporary file for tokenizer loading" << std::endl;
            }
        }
        
        if (verbose_) {
            std::cout << "Checkpoint loaded: " << filename << std::endl;
            std::cout << "  Epochs: " << training_epochs_ << std::endl;
            std::cout << "  Best loss: " << best_loss_ << std::endl;
            std::cout << "  Timestamp: " << training_timestamp_ << std::endl;
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading checkpoint: " << e.what() << std::endl;
        return false;
    }
}

// Other methods remain the same...
void ConversationModel::clear_context() {
    if (context_manager_) {
        context_manager_->clear();
    }
}

void ConversationModel::set_system_prompt(const std::string& prompt) {
    system_prompt_ = prompt;
}

size_t ConversationModel::get_context_token_count() const {
    if (context_manager_) {
        return context_manager_->get_token_count();
    }
    return 0;
}

void ConversationModel::set_tokenizer(std::shared_ptr<BPETokenizer> tokenizer) { 
    tokenizer_ = tokenizer; 
    context_manager_ = std::make_unique<ContextManager>(2048, 20);
    context_manager_->set_tokenizer(tokenizer);
}

} // namespace lm
