#include "lm/models/conversation_model.hpp"
#include "lm/data/training_data.hpp"
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <cereal/archives/binary.hpp>
#include <cereal/types/string.hpp>
#include <limits>

namespace lm {

// Helper function to get current timestamp
std::string ConversationModel::get_current_timestamp() const {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %X");
    return ss.str();
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
    // Initialize with default values
}

// Train method implementation
void ConversationModel::train(const TrainingDataset& dataset, size_t num_epochs, float learning_rate) {
    // Validate tokenizer is set
    if (!tokenizer_) {
        throw std::runtime_error("Tokenizer not set before training");
    }
    
    training_timestamp_ = get_current_timestamp();
    const auto& examples = dataset.examples();
    size_t total_examples = examples.size();
    
    if (verbose_) {
        std::cout << "[" << training_timestamp_ << "] Starting training with " 
                  << total_examples << " examples for " << num_epochs << " epochs" << std::endl;
    }
    
    if (total_examples == 0) {
        std::cerr << "Warning: No training examples provided!" << std::endl;
        return;
    }
    
    // Initialize best loss
    float best_loss = std::numeric_limits<float>::max();
    
    // Training loop
    for (size_t epoch = 0; epoch < num_epochs; epoch++) {
        float epoch_loss = 0.0f;
        size_t examples_processed = 0;
        
        if (verbose_) {
            std::cout << "[" << get_current_timestamp() << "] Epoch " << (epoch + 1) 
                      << "/" << num_epochs << std::endl;
        }
        
        // Process each example
        for (const auto& example : examples) {
            // Encode input and target
            auto input_tokens = tokenizer_->encode(example.input);
            auto target_tokens = tokenizer_->encode(example.target);
            
            if (input_tokens.size() < 1 || target_tokens.size() < 1) {
                continue; // Skip empty sequences
            }
            
            // Training step
            try {
                // Execute training step (returns void)
                transformer_->train_step(input_tokens, target_tokens, learning_rate);
                
                // Calculate loss separately for monitoring
                auto logits = transformer_->forward(input_tokens);
                float loss = transformer_->calculate_loss(logits, target_tokens);
                
                epoch_loss += loss;
                examples_processed++;
                
                if (verbose_ && examples_processed % 100 == 0) {
                    std::cout << "Processed " << examples_processed << " examples, current loss: " << loss << std::endl;
                }
            } catch (const std::exception& e) {
                std::cerr << "Error during training step: " << e.what() << std::endl;
                std::cerr << "Input: '" << example.input << "', Target: '" << example.target << "'" << std::endl;
                // Continue with next example instead of stopping
            }
        }
        
        // Calculate average epoch loss
        if (examples_processed > 0) {
            epoch_loss /= examples_processed;
            
            if (verbose_) {
                std::cout << "Epoch " << (epoch + 1) << " average loss: " << epoch_loss << std::endl;
            }
            
            // Update best loss
            if (epoch_loss < best_loss) {
                best_loss = epoch_loss;
                best_loss_ = best_loss;
                if (verbose_) {
                    std::cout << "New best loss: " << best_loss << std::endl;
                }
                
                // Save best model checkpoint
                save_checkpoint("best_model.checkpoint");
            }
        }
        
        training_epochs_++;
        
        // Learning rate decay
        learning_rate *= 0.95f;
        
        // Periodic checkpoint
        if ((epoch + 1) % 10 == 0) {
            std::string checkpoint_name = "checkpoint_epoch_" + std::to_string(epoch + 1) + ".bin";
            save_checkpoint(checkpoint_name);
            
            if (verbose_) {
                std::cout << "Saved checkpoint: " << checkpoint_name << std::endl;
            }
        }
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
    
    // Encode input
    auto input_tokens = tokenizer_->encode(input);
    
    // Generate response with temperature parameter
    float temperature = 0.8f; // Default temperature value
    auto response_tokens = transformer_->generate(input_tokens, max_response_length_, temperature);
    
    // Decode response
    return tokenizer_->decode(response_tokens);
}

// Save checkpoint method
bool ConversationModel::save_checkpoint(const std::string& filename) {
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs.is_open()) {
        std::cerr << "Failed to open checkpoint file: " << filename << std::endl;
        return false;
    }
    
    try {
        // Create a binary output archive
        cereal::BinaryOutputArchive archive(ofs);
        
        // Serialize the transformer model
        if (transformer_) {
            // Use the transformer's serialize method
            std::stringstream model_stream;
            transformer_->serialize(model_stream);
            std::string model_data = model_stream.str();
            archive(model_data);
        } else {
            std::string empty;
            archive(empty);
        }
        
        // Serialize training metadata
        archive(
            training_epochs_,
            best_loss_,
            training_timestamp_,
            max_response_length_,
            verbose_
        );
        
        // Serialize tokenizer information if available
        if (tokenizer_) {
            // Save tokenizer vocabulary size for verification
            size_t vocab_size = tokenizer_->vocab_size();
            archive(vocab_size);
            
            // Save tokenizer state to a temporary file and read its data
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
                    
                    // Clean up the temporary file
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

// Load checkpoint method
bool ConversationModel::load_checkpoint(const std::string& filename) {
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs.is_open()) {
        std::cerr << "Failed to open checkpoint file: " << filename << std::endl;
        return false;
    }
    
    try {
        // Create a binary input archive
        cereal::BinaryInputArchive archive(ifs);
        
        // Deserialize the transformer model
        std::string model_data;
        archive(model_data);
        
        if (!model_data.empty() && transformer_) {
            std::stringstream model_stream(model_data);
            transformer_->deserialize(model_stream);
        }
        
        // Deserialize training metadata
        archive(
            training_epochs_,
            best_loss_,
            training_timestamp_,
            max_response_length_,
            verbose_
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
                
                // Load tokenizer from the temporary file
                tokenizer_->load(temp_filename);
                
                // Clean up the temporary file
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

// Other methods (context management, system prompt, etc.) would be implemented here
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

} // namespace lm

