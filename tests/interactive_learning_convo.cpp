// interactive_learning_convo.cpp
#include "lm/models/conversation_model.hpp"
#include "lm/tokenizer/bpe_tokenizer.hpp"
#include "lm/data/training_data.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>
#include <memory>
#include <algorithm>
#include <random>
#include <sstream>
#include <fstream>
#include <string>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <unordered_set>
#include <cctype>

// Helper function to get current timestamp
std::string get_current_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %X");
    return ss.str();
}

// Function to validate tokenization
void validate_tokenization(std::shared_ptr<lm::BPETokenizer> tokenizer, const std::string& text) {
    try {
        std::vector<lm::TokenID> tokens = tokenizer->encode(text);
        std::string decoded = tokenizer->decode(tokens);
        
        if (text != decoded) {
            std::cerr << "[" << get_current_timestamp() << "] Tokenization mismatch:" << std::endl;
            std::cerr << "  Original: '" << text << "'" << std::endl;
            std::cerr << "  Decoded: '" << decoded << "'" << std::endl;
            std::cerr << "  Tokens: ";
            for (lm::TokenID id : tokens) {
                std::cerr << id << " ";
            }
            std::cerr << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "[" << get_current_timestamp() << "] Tokenization error: " << e.what() << std::endl;
    }
}

// Enhanced tokenization validation function
void debug_tokenization(std::shared_ptr<lm::BPETokenizer> tokenizer, const std::string& text, const std::string& context = "") {
    try {
        std::vector<lm::TokenID> tokens = tokenizer->encode(text);
        std::string decoded = tokenizer->decode(tokens);
        
        std::cout << "[" << get_current_timestamp() << "] Tokenization " << context << ":" << std::endl;
        std::cout << "  Original: '" << text << "'" << std::endl;
        std::cout << "  Decoded: '" << decoded << "'" << std::endl;
        std::cout << "  Tokens: ";
        for (lm::TokenID id : tokens) {
            std::cout << id << " ";
        }
        std::cout << std::endl;
        std::cout << "  Length: " << tokens.size() << " tokens" << std::endl;
        
        if (text != decoded) {
            std::cerr << "  MISMATCH DETECTED!" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "[" << get_current_timestamp() << "] Tokenization error: " << e.what() << std::endl;
    }
}

// Helper function to trim whitespace
std::string trim(const std::string& str) {
    size_t start = str.find_first_not_of(" \t\n\r\f\v");
    if (start == std::string::npos) return "";
    
    size_t end = str.find_last_not_of(" \t\n\r\f\v");
    return str.substr(start, end - start + 1);
}

// Function to clean and normalize training data
void clean_training_data(const std::string& filename) {
    std::ifstream in_file(filename);
    std::ofstream out_file("cleaned_training_data.txt");
    
    if (!in_file.is_open() || !out_file.is_open()) {
        std::cerr << "[" << get_current_timestamp() << "] ERROR: Could not open files for cleaning!" << std::endl;
        return;
    }
    
    std::string line;
    int skipped = 0;
    int kept = 0;
    
    while (std::getline(in_file, line)) {
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') {
            out_file << line << std::endl;
            continue;
        }
        
        // Skip conversation markers for now
        if (line.find("CONVERSATION:") == 0) {
            out_file << line << std::endl;
            continue;
        }
        
        // Check if this is a single example
        size_t tab_pos = line.find('\t');
        if (tab_pos != std::string::npos) {
            std::string input = line.substr(0, tab_pos);
            std::string target = line.substr(tab_pos + 1);
            
            // Basic cleaning
            if (input.empty() || target.empty()) {
                skipped++;
                continue;
            }
            
            // Remove extra whitespace
            input = trim(input);
            target = trim(target);
            
            // Skip if still empty
            if (input.empty() || target.empty()) {
                skipped++;
                continue;
            }
            
            // Write cleaned example
            out_file << input << "\t" << target << std::endl;
            kept++;
        } else {
            // Just copy the line as-is
            out_file << line << std::endl;
        }
    }
    
    in_file.close();
    out_file.close();
    
    std::cout << "[" << get_current_timestamp() << "] Data cleaning completed. Kept: " << kept << ", Skipped: " << skipped << std::endl;
}

// Function to load training data from a text file
lm::TrainingDataset load_training_dataset(const std::string& filename) {
    lm::TrainingDataset dataset;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "[" << get_current_timestamp() << "] ERROR: Could not open training data file: " << filename << std::endl;
        return dataset;
    }
    
    std::string line;
    std::vector<std::string> current_conversation;
    
    while (std::getline(file, line)) {
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        // Check if this line starts a new conversation
        if (line.find("CONVERSATION:") == 0) {
            if (!current_conversation.empty()) {
                dataset.add_conversation(current_conversation);
                current_conversation.clear();
            }
            continue;
        }
        
        // Check if this is a single example (input and response separated by tab)
        size_t tab_pos = line.find('\t');
        if (tab_pos != std::string::npos) {
            std::string input = line.substr(0, tab_pos);
            std::string target = line.substr(tab_pos + 1);
            dataset.add_example(input, target);
        } else {
            // Add line to current conversation
            current_conversation.push_back(line);
        }
    }
    
    // Add the last conversation if exists
    if (!current_conversation.empty()) {
        dataset.add_conversation(current_conversation);
    }
    
    file.close();
    return dataset;
}

// Function to save new training examples to file
void save_training_example(const std::string& filename, const std::string& input, const std::string& response, bool is_correction = false) {
    std::ofstream file(filename, std::ios_base::app);
    if (file.is_open()) {
        if (is_correction) {
            file << "# Correction added: " << get_current_timestamp() << std::endl;
        } else {
            file << "# New example added: " << get_current_timestamp() << std::endl;
        }
        file << input << "\t" << response << std::endl;
        file.close();
        std::cout << "[" << get_current_timestamp() << "] Saved new training example: " << input << " -> " << response << std::endl;
    } else {
        std::cerr << "[" << get_current_timestamp() << "] ERROR: Could not open training data file for writing: " << filename << std::endl;
    }
}

// Function to save a complete conversation to file
void save_conversation(const std::string& filename, const std::vector<std::string>& conversation) {
    std::ofstream file(filename, std::ios_base::app);
    if (file.is_open()) {
        file << "\n# Conversation added: " << get_current_timestamp() << std::endl;
        file << "CONVERSATION:" << std::endl;
        for (const auto& line : conversation) {
            file << line << std::endl;
        }
        file.close();
        std::cout << "[" << get_current_timestamp() << "] Saved complete conversation with " << conversation.size() << " turns." << std::endl;
    } else {
        std::cerr << "[" << get_current_timestamp() << "] ERROR: Could not open training data file for writing: " << filename << std::endl;
    }
}

// Thread-safe queue for new training examples
class TrainingQueue {
private:
    std::queue<std::pair<std::string, std::string>> queue;
    std::mutex mtx;
    std::condition_variable cv;
    std::atomic<bool> stop_flag{false};
    
public:
    void push(const std::string& input, const std::string& response) {
        std::lock_guard<std::mutex> lock(mtx);
        queue.push({input, response});
        cv.notify_one();
    }
    
    bool pop(std::pair<std::string, std::string>& item) {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [this]() { return !queue.empty() || stop_flag; });
        
        if (stop_flag && queue.empty()) return false;
        
        item = queue.front();
        queue.pop();
        return true;
    }
    
    void stop() {
        stop_flag = true;
        cv.notify_all();
    }
    
    size_t size() {
        std::lock_guard<std::mutex> lock(mtx);
        return queue.size();
    }
};

// Global mutex for model access
std::mutex model_mutex;

// Function to validate matrix dimensions before operations
bool validate_matrix_dimensions(const Eigen::MatrixXf& matrix1, const Eigen::MatrixXf& matrix2) {
    if (matrix1.rows() != matrix2.rows() || matrix1.cols() != matrix2.cols()) {
        std::cerr << "[" << get_current_timestamp() << "] Matrix dimension mismatch: "
                  << matrix1.rows() << "x" << matrix1.cols() << " vs "
                  << matrix2.rows() << "x" << matrix2.cols() << std::endl;
        return false;
    }
    return true;
}

// Function to safely add matrices with dimension validation
bool safe_matrix_add(Eigen::MatrixXf& dst, const Eigen::MatrixXf& src) {
    if (!validate_matrix_dimensions(dst, src)) {
        return false;
    }
    dst += src;
    return true;
}

// Enhanced safe_train function with detailed debugging
void safe_train(lm::ConversationModel& model, std::shared_ptr<lm::BPETokenizer> tokenizer, 
                lm::TrainingDataset& dataset, int epochs, float learning_rate) {
    try {
        std::cout << "[" << get_current_timestamp() << "] Starting safe_train with " 
                  << dataset.size() << " examples" << std::endl;
        
        // Filter out examples with mismatched lengths after tokenization
        lm::TrainingDataset filtered_dataset;
        int skipped_count = 0;
        
        for (const auto& example : dataset.examples()) {
            try {
                auto input_tokens = tokenizer->encode(example.input);
                auto target_tokens = tokenizer->encode(example.target);
                
                std::cout << "[" << get_current_timestamp() << "] Example - Input: '" << example.input 
                          << "' (" << input_tokens.size() << " tokens), Target: '" << example.target 
                          << "' (" << target_tokens.size() << " tokens)" << std::endl;
                
                // Only include examples with reasonable lengths
                if (input_tokens.size() > 0 && target_tokens.size() > 0 && 
                    input_tokens.size() <= 15 && target_tokens.size() <= 15) {
                    filtered_dataset.add_example(example.input, example.target);
                } else {
                    skipped_count++;
                    std::cout << "[" << get_current_timestamp() << "] Skipping example due to length" << std::endl;
                }
            } catch (const std::exception& e) {
                skipped_count++;
                std::cerr << "[" << get_current_timestamp() << "] Error encoding example: " << e.what() << std::endl;
            }
        }
        
        if (skipped_count > 0) {
            std::cout << "[" << get_current_timestamp() << "] Skipped " << skipped_count 
                      << " examples due to length issues" << std::endl;
        }
        
        // Train on filtered dataset
        if (filtered_dataset.size() > 0) {
            std::cout << "[" << get_current_timestamp() << "] Training with " << filtered_dataset.size() 
                      << " examples for " << epochs << " epochs" << std::endl;
            
            // Use a single epoch with very low learning rate
            std::lock_guard<std::mutex> lock(model_mutex);
            
            // Train one example at a time to avoid matrix size issues
            for (const auto& example : filtered_dataset.examples()) {
                try {
                    std::cout << "[" << get_current_timestamp() << "] Training on: '" 
                              << example.input << "' -> '" << example.target << "'" << std::endl;
                    
                    lm::TrainingDataset single_example_dataset;
                    single_example_dataset.add_example(example.input, example.target);
                    model.train(single_example_dataset, 1, 0.0001f);
                    
                    std::cout << "[" << get_current_timestamp() << "] Successfully trained on example" << std::endl;
                } catch (const std::exception& e) {
                    std::cerr << "[" << get_current_timestamp() << "] Error training on example: " 
                              << e.what() << std::endl;
                    std::cerr << "Input: '" << example.input << "', Target: '" << example.target << "'" << std::endl;
                }
            }
        } else {
            std::cout << "[" << get_current_timestamp() << "] No valid examples to train on!" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "[" << get_current_timestamp() << "] Error during training: " << e.what() << std::endl;
    }
}

// Function to validate training examples
bool validate_training_example(std::shared_ptr<lm::BPETokenizer> tokenizer, 
                              const std::string& input, const std::string& target) {
    try {
        auto input_tokens = tokenizer->encode(input);
        auto target_tokens = tokenizer->encode(target);
        
        // Check if tokenization works correctly
        std::string decoded_input = tokenizer->decode(input_tokens);
        std::string decoded_target = tokenizer->decode(target_tokens);
        
        // Basic validation
        if (input_tokens.empty() || target_tokens.empty()) {
            return false;
        }
        
        if (input_tokens.size() > 50 || target_tokens.size() > 50) {
            return false;
        }
        
        // Check if round-trip encoding/decoding works
        if (input != decoded_input || target != decoded_target) {
            return false;
        }
        
        return true;
    } catch (const std::exception& e) {
        return false;
    }
}
// Function to validate model state before training
bool validate_model_for_training(std::shared_ptr<lm::ConversationModel> model) {
    try {
        // Test with a simple input
        std::string test_input = "Hello";
        std::string response;
        {
            std::lock_guard<std::mutex> lock(model_mutex);
            response = model->generate_response(test_input);
        }
        
        // Check if the response is reasonable
        if (response.empty() || response.find("�") != std::string::npos) {
            std::cerr << "[" << get_current_timestamp() << "] Model state validation failed." << std::endl;
            return false;
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[" << get_current_timestamp() << "] Model state validation error: " << e.what() << std::endl;
        return false;
    }
}

// Emergency recovery function
void emergency_recovery(std::shared_ptr<lm::ConversationModel> model, std::shared_ptr<lm::BPETokenizer> tokenizer) {
    try {
        std::cout << "[" << get_current_timestamp() << "] Starting emergency recovery..." << std::endl;
        
        // Try to reload from any available checkpoint
        std::vector<std::string> checkpoints = {
            "continuous_learning_model.bin",
            "final_model.bin",
            "best_model.checkpoint"
        };
        
        bool recovered = false;
        for (const auto& checkpoint : checkpoints) {
            try {
                if (model->load_checkpoint(checkpoint)) {
                    std::cout << "[" << get_current_timestamp() << "] Recovered from checkpoint: " << checkpoint << std::endl;
                    recovered = true;
                    break;
                }
            } catch (const std::exception& e) {
                std::cerr << "[" << get_current_timestamp() << "] Error loading checkpoint " << checkpoint << ": " << e.what() << std::endl;
            }
        }
        
        if (!recovered) {
            std::cout << "[" << get_current_timestamp() << "] No valid checkpoints found, model will need to be retrained" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "[" << get_current_timestamp() << "] Emergency recovery failed: " << e.what() << std::endl;
    }
}

// Function to validate responses
std::string validate_response(const std::string& response) {
    // Check for invalid characters or patterns
    for (char c : response) {
        if (static_cast<unsigned char>(c) > 127 && static_cast<unsigned char>(c) < 192) {
            // This is likely a continuation byte without a start byte
            return "I'm not sure how to respond to that.";
        }
    }
    
    // Check for other patterns that might indicate invalid generation
    if (response.find("�") != std::string::npos) {
        return "I'm not sure how to respond to that.";
    }
    
    // Check if response is just repeating the input
    if (response.length() < 5) {
        return "I'm not sure how to respond to that.";
    }
    
    return response;
}

// Function to process training files
void process_training_file(const std::string& filename, TrainingQueue& training_queue) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[" << get_current_timestamp() << "] ERROR: Could not open training file: " << filename << std::endl;
        return;
    }
    
    std::cout << "[" << get_current_timestamp() << "] Processing training file: " << filename << std::endl;
    
    std::string line;
    std::vector<std::string> current_conversation;
    int examples_added = 0;
    int conversations_added = 0;
    
    while (std::getline(file, line)) {
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        // Check if this line starts a new conversation
        if (line.find("CONVERSATION:") == 0) {
            // Process the current conversation if it exists
            if (current_conversation.size() >= 2) {
                // Add conversation turns as training examples
                for (size_t i = 0; i < current_conversation.size() - 1; i += 2) {
                    if (i + 1 < current_conversation.size()) {
                        training_queue.push(current_conversation[i], current_conversation[i + 1]);
                        examples_added++;
                    }
                }
                conversations_added++;
            }
            current_conversation.clear();
            continue;
        }
        
        // Check if this is a single example (input and response separated by tab)
        size_t tab_pos = line.find('\t');
        if (tab_pos != std::string::npos) {
            std::string input = line.substr(0, tab_pos);
            std::string target = line.substr(tab_pos + 1);
            training_queue.push(input, target);
            examples_added++;
        } else {
            // Add line to current conversation
            current_conversation.push_back(line);
        }
    }
    
    // Process the last conversation if exists
    if (current_conversation.size() >= 2) {
        for (size_t i = 0; i < current_conversation.size() - 1; i += 2) {
            if (i + 1 < current_conversation.size()) {
                training_queue.push(current_conversation[i], current_conversation[i + 1]);
                examples_added++;
            }
        }
        conversations_added++;
    }
    
    file.close();
    std::cout << "[" << get_current_timestamp() << "] Added " << examples_added << " examples from " 
              << conversations_added << " conversations to training queue." << std::endl;
}

// Function to create a minimal dataset for stability testing
void create_minimal_dataset() {
    std::ofstream file("minimal_training.txt");
    if (file.is_open()) {
        file << "# Minimal training dataset for stability testing\n";
        file << "Hello\tHi\n";
        file << "Hi\tHello\n";
        file << "Hey\tHey\n";
        file << "Yes\tYes\n";
        file << "No\tNo\n";
        file << "Why\tWhy\n";
        file << "What\tWhat\n";
        file.close();
        std::cout << "[" << get_current_timestamp() << "] Created minimal training dataset." << std::endl;
    }
}

// Function to validate the model state
bool validate_model_state(std::shared_ptr<lm::ConversationModel> model) {
    try {
        // Test with a simple input
        std::string test_input = "Hello";
        std::string response;
        {
            std::lock_guard<std::mutex> lock(model_mutex);
            response = model->generate_response(test_input);
        }
        
        // Check if the response is reasonable
        if (response.empty() || response.find("�") != std::string::npos) {
            std::cerr << "[" << get_current_timestamp() << "] Model state validation failed." << std::endl;
            return false;
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[" << get_current_timestamp() << "] Model state validation error: " << e.what() << std::endl;
        return false;
    }
}

// Function to recover from training errors
bool recover_from_training_error(std::shared_ptr<lm::ConversationModel> model, std::shared_ptr<lm::BPETokenizer> tokenizer) {
    try {
        // Try to reload the model from a known good state
        std::vector<std::string> checkpoints = {
            "continuous_learning_model.bin",
            "final_model.bin"
        };
        
        for (const auto& checkpoint : checkpoints) {
            if (model->load_checkpoint(checkpoint)) {
                std::cout << "[" << get_current_timestamp() << "] Recovered model from checkpoint: " << checkpoint << std::endl;
                return true;
            }
        }
        
        std::cout << "[" << get_current_timestamp() << "] No valid checkpoints found." << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "[" << get_current_timestamp() << "] Recovery failed: " << e.what() << std::endl;
        return false;
    }
}

// Fallback training function for when standard training fails
void fallback_train(lm::ConversationModel& model, std::shared_ptr<lm::BPETokenizer> tokenizer, 
                   lm::TrainingDataset& dataset, int epochs, float learning_rate) {
    try {
        std::cout << "[" << get_current_timestamp() << "] Starting fallback training mode" << std::endl;
        
        // Use only very simple examples
        lm::TrainingDataset simple_dataset;
        
        for (const auto& example : dataset.examples()) {
            try {
                auto input_tokens = tokenizer->encode(example.input);
                auto target_tokens = tokenizer->encode(example.target);
                
                // Only include very short examples with exact length matching
                if (input_tokens.size() == 1 && target_tokens.size() == 1) {
                    simple_dataset.add_example(example.input, example.target);
                    std::cout << "[" << get_current_timestamp() << "] Using simple example: '" 
                              << example.input << "' -> '" << example.target << "'" << std::endl;
                }
            } catch (const std::exception& e) {
                // Skip examples that cause errors
            }
        }
        
        if (simple_dataset.size() > 0) {
            std::cout << "[" << get_current_timestamp() << "] Fallback training with " 
                      << simple_dataset.size() << " simple examples" << std::endl;
            
            std::lock_guard<std::mutex> lock(model_mutex);
            
            // Use an extremely low learning rate
            model.train(simple_dataset, 1, 0.00001f);
        } else {
            std::cout << "[" << get_current_timestamp() << "] No simple examples available for fallback training" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "[" << get_current_timestamp() << "] Error during fallback training: " << e.what() << std::endl;
        throw; // Re-throw to let the caller handle it
    }
}

bool check_matrix_dimensions_compatible(std::shared_ptr<lm::BPETokenizer> tokenizer, 
                                       const lm::TrainingDataset& dataset) {
    try {
        for (const auto& example : dataset.examples()) {
            auto input_tokens = tokenizer->encode(example.input);
            auto target_tokens = tokenizer->encode(example.target);
            
            // Check if token lengths are reasonable and compatible
            if (input_tokens.empty() || target_tokens.empty()) {
                return false;
            }
            
            // For transformer models, we need to ensure sequence lengths are compatible
            // with the model's expected dimensions
            if (input_tokens.size() > 100 || target_tokens.size() > 100) {
                return false;
            }
            
            // Additional checks can be added here based on your model's specific requirements
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[" << get_current_timestamp() << "] Error checking matrix dimensions: " << e.what() << std::endl;
        return false;
    }
}

// Ultra-safe training function that avoids matrix dimension issues
void ultra_safe_train(lm::ConversationModel& model, std::shared_ptr<lm::BPETokenizer> tokenizer, 
                     lm::TrainingDataset& dataset, int epochs, float learning_rate) {
    try {
        std::cout << "[" << get_current_timestamp() << "] Starting ultra-safe training with " 
                  << dataset.size() << " examples" << std::endl;
        
        // Check if matrix dimensions are compatible before training
        if (!check_matrix_dimensions_compatible(tokenizer, dataset)) {
            std::cerr << "[" << get_current_timestamp() << "] Matrix dimensions incompatible, skipping training" << std::endl;
            return;
        }
        
        // Use a single epoch with very low learning rate
        std::lock_guard<std::mutex> lock(model_mutex);
        
        // Train one example at a time with additional validation
        for (const auto& example : dataset.examples()) {
            try {
                // Double-check this specific example
                auto input_tokens = tokenizer->encode(example.input);
                auto target_tokens = tokenizer->encode(example.target);
                
                if (input_tokens.empty() || target_tokens.empty() ||
                    input_tokens.size() > 50 || target_tokens.size() > 50) {
                    std::cerr << "[" << get_current_timestamp() << "] Skipping problematic example: '" 
                              << example.input << "' -> '" << example.target << "'" << std::endl;
                    continue;
                }
                
                lm::TrainingDataset single_example_dataset;
                single_example_dataset.add_example(example.input, example.target);
                
                // Use an extremely conservative learning rate
                model.train(single_example_dataset, 1, 0.00001f);
                
            } catch (const std::exception& e) {
                std::cerr << "[" << get_current_timestamp() << "] Error training on example: " 
                          << e.what() << std::endl;
                std::cerr << "Input: '" << example.input << "', Target: '" << example.target << "'" << std::endl;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[" << get_current_timestamp() << "] Error during ultra-safe training: " << e.what() << std::endl;
    }
}

// Training thread function with ultra-safe training
void training_thread_func(
    std::shared_ptr<lm::ConversationModel> model,
    std::shared_ptr<lm::BPETokenizer> tokenizer,
    TrainingQueue& training_queue,
    const std::string& training_data_file,
    std::atomic<bool>& stop_training,
    std::atomic<int>& training_round,
    std::atomic<bool>& training_complete
) {
    try {
        std::cout << "[" << get_current_timestamp() << "] Training thread started" << std::endl;
        
        while (!stop_training) {
            // Check for new training examples
            std::pair<std::string, std::string> example;
            int examples_processed = 0;
            
            // Process up to 3 examples at a time (reduced from 5)
            while (training_queue.pop(example) && examples_processed < 3 && !stop_training) {
                // Validate the example before saving
                if (validate_training_example(tokenizer, example.first, example.second)) {
                    save_training_example(training_data_file, example.first, example.second);
                    examples_processed++;
                } else {
                    std::cerr << "[" << get_current_timestamp() << "] Skipping invalid training example" << std::endl;
                }
            }
            
            if (examples_processed > 0 && !stop_training) {
                // Reload the training dataset
                auto dataset = load_training_dataset(training_data_file);
                
                if (dataset.size() > 0 && !stop_training) {
                    std::cout << "[" << get_current_timestamp() << "] Background training with " << dataset.size() << " examples..." << std::endl;
                    
                    // Validate model state before training
                    if (!validate_model_state(model)) {
                        std::cerr << "[" << get_current_timestamp() << "] Model is in an invalid state, attempting recovery..." << std::endl;
                        emergency_recovery(model, tokenizer);
                        
                        // Skip training if recovery failed
                        if (!validate_model_state(model)) {
                            std::cerr << "[" << get_current_timestamp() << "] Recovery failed, skipping training" << std::endl;
                            continue;
                        }
                    }
                    
                    // Use ultra-safe training instead of safe_train or fallback_train
                    try {
                        std::cout << "[" << get_current_timestamp() << "] Attempting ultra-safe training..." << std::endl;
                        ultra_safe_train(*model, tokenizer, dataset, 1, 0.00001f);
                        training_round++;
                        std::cout << "[" << get_current_timestamp() << "] Ultra-safe training completed. Training round: " << training_round << std::endl;
                    } catch (const std::exception& e) {
                        std::cerr << "[" << get_current_timestamp() << "] Ultra-safe training failed: " << e.what() << std::endl;
                        
                        // Attempt emergency recovery
                        emergency_recovery(model, tokenizer);
                    }
                    
                    // Save model checkpoint after successful training
                    try {
                        std::lock_guard<std::mutex> lock(model_mutex);
                        if (model->save_checkpoint("continuous_learning_model.bin")) {
                            std::cout << "[" << get_current_timestamp() << "] Saved model checkpoint after training" << std::endl;
                        }
                    } catch (const std::exception& e) {
                        std::cerr << "[" << get_current_timestamp() << "] Error saving checkpoint: " << e.what() << std::endl;
                    }
                }
            }
            
            // Sleep for a while before checking again, but check stop flag frequently
            for (int i = 0; i < 60 && !stop_training; i++) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[" << get_current_timestamp() << "] FATAL ERROR in training thread: " << e.what() << std::endl;
    }
    
    training_complete = true;
    std::cout << "[" << get_current_timestamp() << "] Training thread stopped" << std::endl;
}

// Function to check if a response needs improvement
bool needs_improvement(const std::string& response) {
    // Simple heuristic: very short or generic responses might need improvement
    if (response.length() < 5) return true;
    
    // Check for generic responses that don't provide much value
    std::vector<std::string> generic_responses = {
        "I don't know", "I'm not sure", "I can't answer", "I don't understand",
        "That's interesting", "Tell me more", "I see", "Okay", "Right"
    };
    
    for (const auto& generic : generic_responses) {
        if (response.find(generic) == 0) {
            return true;
        }
    }
    
    return false;
}

// Function to process commands with '/' prefix
bool process_command(
    const std::string& input, 
    std::vector<std::string>& current_conversation,
    TrainingQueue& training_queue,
    std::atomic<int>& training_round,
    std::atomic<bool>& shutdown_requested,
    std::atomic<bool>& stop_training,
    std::shared_ptr<lm::ConversationModel> model,
    std::shared_ptr<lm::BPETokenizer> tokenizer
) {
    if (input.empty() || input[0] != '/') {
        return false;
    }
    
    // Extract command and arguments
    std::string command;
    std::string argument;
    size_t space_pos = input.find(' ');
    
    if (space_pos != std::string::npos) {
        command = input.substr(1, space_pos - 1);
        argument = input.substr(space_pos + 1);
    } else {
        command = input.substr(1);
    }
    
    // Convert command to lowercase for case-insensitive comparison
    std::transform(command.begin(), command.end(), command.begin(), ::tolower);
    
    // Process commands
    if (command == "quit" || command == "exit") {
        std::cout << "[" << get_current_timestamp() << "] Shutting down..." << std::endl;
        stop_training = true;
        shutdown_requested = true;
        return true;
    }
    else if (command == "save" && !current_conversation.empty()) {
        save_conversation("training_data.txt", current_conversation);
        current_conversation.clear();
        return true;
    }
    else if (command == "help") {
        std::cout << "Available commands:" << std::endl;
        std::cout << "  /help - Show this help message" << std::endl;
        std::cout << "  /quit or /exit - Shutdown the application" << std::endl;
        std::cout << "  /save - Save the current conversation to training data" << std::endl;
        std::cout << "  /train - Show training status" << std::endl;
        std::cout << "  /correct <response> - Correct the last AI response" << std::endl;
        std::cout << "  /clear - Clear the current conversation history" << std::endl;
        std::cout << "  /status - Show system status" << std::endl;
        std::cout << "  /emergency_stop - Immediately stop training and shutdown" << std::endl;
        std::cout << "  /trainfile <filename> - Load and train on conversations from a file" << std::endl;
        std::cout << "  /reset_model - Reset the model to a known good state" << std::endl;
        return true;
    }
    else if (command == "train") {
        std::cout << "Training round: " << training_round << std::endl;
        std::cout << "Pending training examples: " << training_queue.size() << std::endl;
        return true;
    }
    else if (command == "correct" && !current_conversation.empty()) {
        // Handle correction of the last response
        if (argument.empty()) {
            std::cout << "Usage: /correct <your better response>" << std::endl;
            return true;
        }
        
        if (current_conversation.size() >= 2) {
            std::string last_user_input = current_conversation[current_conversation.size() - 2];
            training_queue.push(last_user_input, argument);
            std::cout << "Correction saved. The AI will learn from this." << std::endl;
            
            // Replace the last response in the conversation
            current_conversation.back() = argument;
            std::cout << "AI: " << argument << std::endl;
        }
        return true;
    }
    else if (command == "clear") {
        current_conversation.clear();
        std::cout << "Conversation history cleared." << std::endl;
        return true;
    }
    else if (command == "status") {
        std::cout << "System Status:" << std::endl;
        std::cout << "  Training rounds completed: " << training_round << std::endl;
        std::cout << "  Pending training examples: " << training_queue.size() << std::endl;
        std::cout << "  Conversation history size: " << current_conversation.size() << " turns" << std::endl;
        return true;
    }
    else if (command == "emergency_stop") {
        std::cout << "[" << get_current_timestamp() << "] EMERGENCY SHUTDOWN INITIATED!" << std::endl;
        stop_training = true;
        shutdown_requested = true;
        return true;
    }
    else if (command == "trainfile") {
        if (argument.empty()) {
            std::cout << "Usage: /trainfile <filename>" << std::endl;
        } else {
            process_training_file(argument, training_queue);
        }
        return true;
    }
    else if (command == "reset_model") {
        std::cout << "[" << get_current_timestamp() << "] Resetting model..." << std::endl;
        if (recover_from_training_error(model, tokenizer)) {
            std::cout << "Model reset successfully." << std::endl;
        } else {
            std::cout << "Model reset failed. You may need to restart the application." << std::endl;
        }
        return true;
    }
    else {
        std::cout << "Unknown command: " << command << std::endl;
        std::cout << "Type /help for available commands." << std::endl;
        return true;
    }
    
    return false;
}

// Interactive conversation function
void interactive_conversation(
    std::shared_ptr<lm::ConversationModel> model,
    TrainingQueue& training_queue,
    std::atomic<int>& training_round,
    std::atomic<bool>& shutdown_requested,
    std::atomic<bool>& stop_training,
    std::shared_ptr<lm::BPETokenizer> tokenizer
) {
    std::cout << "[" << get_current_timestamp() << "] Starting interactive conversation mode..." << std::endl;
    std::cout << "Type /help for available commands, /quit to exit." << std::endl;
    std::cout << "==========================================" << std::endl;
    
    std::vector<std::string> current_conversation;
    std::string input;
    int conversation_turns = 0;
    
    while (!shutdown_requested) {
        std::cout << "You: ";
        std::getline(std::cin, input);
        
        // Check for commands
        if (process_command(input, current_conversation, training_queue, training_round, 
                           shutdown_requested, stop_training, model, tokenizer)) {
            if (shutdown_requested) {
                break;
            }
            continue;
        }
        
        // Skip empty input
        if (input.empty()) {
            continue;
        }
        
        // Add user input to conversation
        current_conversation.push_back(input);
        conversation_turns++;
        
        // Generate response
        std::string response;
        {
            std::lock_guard<std::mutex> lock(model_mutex);
            response = model->generate_response(input);
        }
        
        // Validate the response
        response = validate_response(response);
        
        // Add AI response to conversation
        current_conversation.push_back(response);
        conversation_turns++;
        
        // Display response
        std::cout << "AI: " << response << std::endl;
        
        // Check if this exchange should be added to training data
        if (conversation_turns >= 2) {
            // Add to training queue for background learning
            training_queue.push(input, response);
            
            // If the response seems weak, prompt for improvement
            if (needs_improvement(response)) {
                std::cout << "[[ Would you like to provide a better response? Type '/correct <your better response>' or just continue. ]]" << std::endl;
            }
        }
        
        // Limit conversation history to prevent excessive memory usage
        if (current_conversation.size() > 20) {
            current_conversation.erase(current_conversation.begin(), current_conversation.begin() + 2);
        }
    }
    
    // Save the conversation if it has meaningful content
    if (conversation_turns >= 4 && !shutdown_requested) {
        std::cout << "Save this conversation? (y/n): ";
        std::string answer;
        std::getline(std::cin, answer);
        if (answer == "y" || answer == "yes") {
            save_conversation("training_data.txt", current_conversation);
        }
    }
}

int main() {
    std::cout << "[" << get_current_timestamp() << "] Starting continuous learning conversation system..." << std::endl;
    
    try {
        // Create a minimal dataset for stability testing
        create_minimal_dataset();
        
        // Initialize tokenizer
        std::cout << "[" << get_current_timestamp() << "] Creating BPE tokenizer..." << std::endl;
        auto tokenizer = std::make_shared<lm::BPETokenizer>();
        
        // Clean the training data first
        clean_training_data("training_data.txt");
        
        // Load the cleaned training data
        std::cout << "[" << get_current_timestamp() << "] Loading training data from file..." << std::endl;
        auto training_dataset = load_training_dataset("cleaned_training_data.txt");
        
        if (training_dataset.size() == 0) {
            std::cerr << "[" << get_current_timestamp() << "] ERROR: No training data loaded!" << std::endl;
            return 1;
        }
        
        // Only use a subset of the data for initial testing
        if (training_dataset.size() > 100) {
            lm::TrainingDataset small_dataset;
            for (size_t i = 0; i < 100; i++) {
                small_dataset.add_example(training_dataset.examples()[i].input, 
                                         training_dataset.examples()[i].target);
            }
            training_dataset = small_dataset;
            std::cout << "Using reduced dataset of " << training_dataset.size() << " examples for testing." << std::endl;
        }
        
        std::cout << "[" << get_current_timestamp() << "] Loaded " << training_dataset.size() << " training examples." << std::endl;
        
        // Create comprehensive tokenizer training data
        std::cout << "[" << get_current_timestamp() << "] Preparing training data for tokenizer..." << std::endl;
        std::vector<std::string> tokenizer_training_data;
        
        // Add general text data for tokenizer
        std::vector<std::string> general_texts = {
            "Hello, how are you?", "I'm doing well, thank you!", "What can I help you with today?",
            "The weather is nice today.", "I enjoy programming in C++.", "Machine learning is fascinating.",
            "Natural language processing enables computers to understand human language.",
            "This is a test of the tokenizer system.", "Reinforcement learning uses rewards to train agents.",
            "Deep learning models have many layers.", "The quick brown fox jumps over the lazy dog.",
            "Artificial intelligence is transforming the world.", "Machine learning enables computers to learn from data.",
            "Natural language processing helps computers understand human language.",
            "Conversational AI is becoming increasingly sophisticated.",
            "Transformers have revolutionized natural language processing.",
            "Attention mechanisms allow models to focus on relevant information.",
            "Tokenization is the first step in many NLP pipelines.",
            "Word embeddings represent words as vectors in high-dimensional space.",
            "Language models can generate human-like text."
        };
        
        for (const auto& text : general_texts) {
            tokenizer_training_data.push_back(text);
        }
        
        // Add data from our conversation dataset
        for (const auto& example : training_dataset.examples()) {
            tokenizer_training_data.push_back(example.input);
            tokenizer_training_data.push_back(example.target);
        }
        
        std::cout << "[" << get_current_timestamp() << "] Training tokenizer with " 
                  << tokenizer_training_data.size() << " examples..." << std::endl;
        
        // Train with a more reasonable vocabulary size
        tokenizer->train(tokenizer_training_data, 500);  // Reduced from 1000 to 500
        
        std::cout << "[" << get_current_timestamp() << "] Tokenizer training completed. Vocabulary size: " 
                  << tokenizer->vocab_size() << std::endl;
        
        // Add more validation
        debug_tokenization(tokenizer, "Hello", "simple greeting");
        debug_tokenization(tokenizer, "Hello.", "greeting with punctuation");
        debug_tokenization(tokenizer, "What is your name?", "question");
        debug_tokenization(tokenizer, "I'm an AI assistant.", "response");
        
        // Initialize conversation model with smaller parameters
        std::cout << "[" << get_current_timestamp() << "] Initializing conversation model..." << std::endl;
        auto model = std::make_shared<lm::ConversationModel>(
            tokenizer->vocab_size(), // vocab_size
            128,                     // d_model (reduced from 512)
            2,                       // n_layers (reduced from 6)
            2,                       // n_heads (reduced from 8)
            512,                     // d_ff (reduced from 2048)
            0.1f                     // dropout
        );
        model->set_tokenizer(tokenizer);
        model->set_verbose(false);    // Disable verbose output for interactive use
        model->set_max_response_length(50); // Allow longer responses
        
        // Try to load a previous model checkpoint if available
        bool model_loaded = false;
        std::vector<std::string> checkpoints = {
            "continuous_learning_model.bin",
            "final_model.bin"
        };
        
        for (const auto& checkpoint : checkpoints) {
            if (model->load_checkpoint(checkpoint)) {
                std::cout << "[" << get_current_timestamp() << "] Loaded model from checkpoint: " << checkpoint << std::endl;
                model_loaded = true;
                break;
            }
        }
        
        // If no checkpoint was loaded, train the model from scratch
        if (!model_loaded) {
            std::cout << "[" << get_current_timestamp() << "] Training conversation model with " 
                      << training_dataset.size() << " examples..." << std::endl;
            
            // Use safe training function with reduced parameters
            safe_train(*model, tokenizer, training_dataset, 5, 0.005f); // Reduced epochs and learning rate
            
            // Save initial model
            if (model->save_checkpoint("continuous_learning_model.bin")) {
                std::cout << "[" << get_current_timestamp() << "] Saved initial model." << std::endl;
            }
        }
        
        // Validate model state before starting training thread
        if (!validate_model_state(model)) {
            std::cerr << "[" << get_current_timestamp() << "] Model is in an invalid state. Attempting recovery..." << std::endl;
            if (!recover_from_training_error(model, tokenizer)) {
                std::cerr << "[" << get_current_timestamp() << "] Recovery failed. Exiting." << std::endl;
                return 1;
            }
        }
        
        // Set up continuous learning system
        TrainingQueue training_queue;
        std::atomic<bool> stop_training{false};
        std::atomic<bool> training_complete{false};
        std::atomic<int> training_round{0};
        std::atomic<bool> shutdown_requested{false};
        
        // Start background training thread
        std::thread training_thread(
            training_thread_func, 
            model, 
            tokenizer, 
            std::ref(training_queue), 
            "training_data.txt",
            std::ref(stop_training),
            std::ref(training_round),
            std::ref(training_complete)
        );
        
        // Run interactive conversation
        interactive_conversation(model, training_queue, training_round, shutdown_requested, stop_training, tokenizer);
        
        // Stop the training thread gracefully
        std::cout << "[" << get_current_timestamp() << "] Stopping training thread..." << std::endl;
        stop_training = true;
        training_queue.stop();
        
        // Wait for training thread to complete
        if (training_thread.joinable()) {
            // Wait for a reasonable amount of time
            auto start = std::chrono::steady_clock::now();
            while (!training_complete && 
                   std::chrono::steady_clock::now() - start < std::chrono::seconds(10)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            
            if (training_thread.joinable()) {
                if (training_complete) {
                    training_thread.join();
                    std::cout << "[" << get_current_timestamp() << "] Training thread stopped gracefully." << std::endl;
                } else {
                    std::cerr << "[" << get_current_timestamp() << "] WARNING: Training thread did not complete in time. Forcing shutdown." << std::endl;
                    training_thread.detach();
                }
            }
        }
        
        // Save the final model
        {
            std::lock_guard<std::mutex> lock(model_mutex);
            if (model->save_checkpoint("continuous_learning_model.bin")) {
                std::cout << "[" << get_current_timestamp() << "] Saved final model with continuous learning." << std::endl;
            }
        }
        
        std::cout << "[" << get_current_timestamp() << "] Continuous learning system shutdown complete." << std::endl;
        std::cout << "Total training rounds completed: " << training_round << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "[" << get_current_timestamp() << "] Error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "[" << get_current_timestamp() << "] Unknown error occurred." << std::endl;
        return 1;
    }
    
    return 0;
}
