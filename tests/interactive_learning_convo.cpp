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
#include <csignal>
#include <functional>
#include <cstddef>

// ============================================================================
// Utility Functions
// ============================================================================

std::string get_current_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %X");
    return ss.str();
}

std::string trim(const std::string& str) {
    size_t start = str.find_first_not_of(" \t\n\r\f\v");
    if (start == std::string::npos) return "";
    
    size_t end = str.find_last_not_of(" \t\n\r\f\v");
    return str.substr(start, end - start + 1);
}

// ============================================================================
// Signal Handler for Eigen Assertions
// ============================================================================

class SignalHandler {
private:
    static std::function<void(int)> current_handler_;
    static struct sigaction old_action_;
    
public:
    static void set_assertion_handler(std::function<void(int)> handler) {
        current_handler_ = handler;
        
        struct sigaction new_action;
        new_action.sa_handler = signal_handler;
        sigemptyset(&new_action.sa_mask);
        new_action.sa_flags = 0;
        
        sigaction(SIGABRT, &new_action, &old_action_);
    }
    
    static void restore_original_handler() {
        sigaction(SIGABRT, &old_action_, nullptr);
    }
    
private:
    static void signal_handler(int signal) {
        if (signal == SIGABRT && current_handler_) {
            current_handler_(signal);
        }
        restore_original_handler();
        raise(signal);
    }
};

std::function<void(int)> SignalHandler::current_handler_ = nullptr;
struct sigaction SignalHandler::old_action_ = {};

// ============================================================================
// Training Bypass System
// ============================================================================

class TrainingBypass {
private:
    std::shared_ptr<lm::BPETokenizer> tokenizer_;
    int consecutive_failures_ = 0;
    const int MAX_CONSECUTIVE_FAILURES = 3;
    
public:
    TrainingBypass(std::shared_ptr<lm::BPETokenizer> tokenizer) : tokenizer_(tokenizer) {}
    
    bool should_bypass_training(const lm::TrainingDataset& dataset) {
        if (consecutive_failures_ >= MAX_CONSECUTIVE_FAILURES) {
            std::cerr << "[" << get_current_timestamp() << "] Training bypassed due to " 
                      << consecutive_failures_ << " consecutive failures" << std::endl;
            return true;
        }
        return !validate_dataset_compatibility(dataset);
    }
    
    void report_training_failure() {
        consecutive_failures_++;
        std::cerr << "[" << get_current_timestamp() << "] Training failure reported. Consecutive failures: " 
                  << consecutive_failures_ << std::endl;
    }
    
    void report_training_success() {
        consecutive_failures_ = 0;
    }
    
    void reset() {
        consecutive_failures_ = 0;
    }
    
private:
    bool validate_dataset_compatibility(const lm::TrainingDataset& dataset) {
        try {
            for (const auto& example : dataset.examples()) {
                auto input_tokens = tokenizer_->encode(example.input);
                auto target_tokens = tokenizer_->encode(example.target);
                
                if (input_tokens.empty() || target_tokens.empty()) {
                    std::cerr << "[" << get_current_timestamp() << "] Empty tokens detected" << std::endl;
                    return false;
                }
                
                if (input_tokens.size() > 50 || target_tokens.size() > 50) {
                    std::cerr << "[" << get_current_timestamp() << "] Sequence too long: " 
                              << input_tokens.size() << " / " << target_tokens.size() << std::endl;
                    return false;
                }
                
                std::string decoded_input = tokenizer_->decode(input_tokens);
                std::string decoded_target = tokenizer_->decode(target_tokens);
                
                if (example.input != decoded_input || example.target != decoded_target) {
                    std::cerr << "[" << get_current_timestamp() << "] Tokenization round-trip failed" << std::endl;
                    return false;
                }
            }
            return true;
        } catch (const std::exception& e) {
            std::cerr << "[" << get_current_timestamp() << "] Dataset validation error: " << e.what() << std::endl;
            return false;
        }
    }
};

// ============================================================================
// Safe Training Wrapper
// ============================================================================

class SafeTrainingWrapper {
private:
    std::shared_ptr<lm::ConversationModel> model_;
    TrainingBypass bypass_;
    mutable std::mutex training_mutex_;
    
public:
    SafeTrainingWrapper(std::shared_ptr<lm::ConversationModel> model, 
                       std::shared_ptr<lm::BPETokenizer> tokenizer)
        : model_(model), bypass_(tokenizer) {}
    
    bool safe_train(const lm::TrainingDataset& dataset, int epochs = 1, float learning_rate = 0.0001f) {
        std::lock_guard<std::mutex> lock(training_mutex_);
        
        if (bypass_.should_bypass_training(dataset)) {
            std::cerr << "[" << get_current_timestamp() << "] Training bypassed" << std::endl;
            return false;
        }
        
        bool training_successful = false;
        bool assertion_caught = false;
        
        // Set up signal handler to catch assertion failures
        SignalHandler::set_assertion_handler([&training_successful, &assertion_caught](int signal) {
            std::cerr << "[" << get_current_timestamp() << "] Caught assertion failure signal: " << signal << std::endl;
            training_successful = false;
            assertion_caught = true;
        });
        
        try {
            model_->train(dataset, epochs, learning_rate);
            training_successful = true;
        } catch (const std::exception& e) {
            std::cerr << "[" << get_current_timestamp() << "] Training exception: " << e.what() << std::endl;
            training_successful = false;
        }
        
        SignalHandler::restore_original_handler();
        
        if (training_successful && !assertion_caught) {
            bypass_.report_training_success();
            return true;
        } else {
            bypass_.report_training_failure();
            return false;
        }
    }
    
    bool alternative_train(const lm::TrainingDataset& dataset) {
        std::lock_guard<std::mutex> lock(training_mutex_);
        
        try {
            if (dataset.size() > 0) {
                // Train on just the first example
                lm::TrainingDataset single_example;
                single_example.add_example(dataset.examples()[0].input, dataset.examples()[0].target);
                
                model_->train(single_example, 1, 0.00001f);
                bypass_.report_training_success();
                return true;
            }
            return false;
        } catch (const std::exception& e) {
            std::cerr << "[" << get_current_timestamp() << "] Alternative training failed: " << e.what() << std::endl;
            bypass_.report_training_failure();
            return false;
        }
    }
    
    void reset_bypass() {
        bypass_.reset();
    }
};

// ============================================================================
// Thread-safe Training Queue
// ============================================================================

class TrainingQueue {
private:
    std::queue<std::pair<std::string, std::string>> queue_;
    std::mutex mtx_;
    std::condition_variable cv_;
    std::atomic<bool> stop_flag_{false};
    
public:
    void push(const std::string& input, const std::string& response) {
        std::lock_guard<std::mutex> lock(mtx_);
        queue_.push({input, response});
        cv_.notify_one();
    }
    
    bool pop(std::pair<std::string, std::string>& item) {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [this]() { return !queue_.empty() || stop_flag_; });
        
        if (stop_flag_ && queue_.empty()) return false;
        
        item = queue_.front();
        queue_.pop();
        return true;
    }
    
    void stop() {
        stop_flag_ = true;
        cv_.notify_all();
    }
    
    size_t size() {
        std::lock_guard<std::mutex> lock(mtx_);
        return queue_.size();
    }
};

// ============================================================================
// Data Processing Functions
// ============================================================================

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
        if (line.empty() || line[0] == '#') {
            out_file << line << std::endl;
            continue;
        }
        
        if (line.find("CONVERSATION:") == 0) {
            out_file << line << std::endl;
            continue;
        }
        
        size_t tab_pos = line.find('\t');
        if (tab_pos != std::string::npos) {
            std::string input = line.substr(0, tab_pos);
            std::string target = line.substr(tab_pos + 1);
            
            if (input.empty() || target.empty()) {
                skipped++;
                continue;
            }
            
            input = trim(input);
            target = trim(target);
            
            if (input.empty() || target.empty()) {
                skipped++;
                continue;
            }
            
            out_file << input << "\t" << target << std::endl;
            kept++;
        } else {
            out_file << line << std::endl;
        }
    }
    
    in_file.close();
    out_file.close();
    
    std::cout << "[" << get_current_timestamp() << "] Data cleaning completed. Kept: " << kept << ", Skipped: " << skipped << std::endl;
}

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
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        if (line.find("CONVERSATION:") == 0) {
            if (!current_conversation.empty()) {
                dataset.add_conversation(current_conversation);
                current_conversation.clear();
            }
            continue;
        }
        
        size_t tab_pos = line.find('\t');
        if (tab_pos != std::string::npos) {
            std::string input = line.substr(0, tab_pos);
            std::string target = line.substr(tab_pos + 1);
            dataset.add_example(input, target);
        } else {
            current_conversation.push_back(line);
        }
    }
    
    if (!current_conversation.empty()) {
        dataset.add_conversation(current_conversation);
    }
    
    file.close();
    return dataset;
}

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
        std::cout << "[" << get_current_timestamp() << "] Saved training example: " << input << " -> " << response << std::endl;
    } else {
        std::cerr << "[" << get_current_timestamp() << "] ERROR: Could not open training data file for writing: " << filename << std::endl;
    }
}

void save_conversation(const std::string& filename, const std::vector<std::string>& conversation) {
    std::ofstream file(filename, std::ios_base::app);
    if (file.is_open()) {
        file << "\n# Conversation added: " << get_current_timestamp() << std::endl;
        file << "CONVERSATION:" << std::endl;
        for (const auto& line : conversation) {
            file << line << std::endl;
        }
        file.close();
        std::cout << "[" << get_current_timestamp() << "] Saved conversation with " << conversation.size() << " turns" << std::endl;
    } else {
        std::cerr << "[" << get_current_timestamp() << "] ERROR: Could not open training data file for writing: " << filename << std::endl;
    }
}

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
    
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        if (line.find("CONVERSATION:") == 0) {
            if (current_conversation.size() >= 2) {
                for (size_t i = 0; i < current_conversation.size() - 1; i += 2) {
                    if (i + 1 < current_conversation.size()) {
                        training_queue.push(current_conversation[i], current_conversation[i + 1]);
                        examples_added++;
                    }
                }
            }
            current_conversation.clear();
            continue;
        }
        
        size_t tab_pos = line.find('\t');
        if (tab_pos != std::string::npos) {
            std::string input = line.substr(0, tab_pos);
            std::string target = line.substr(tab_pos + 1);
            training_queue.push(input, target);
            examples_added++;
        } else {
            current_conversation.push_back(line);
        }
    }
    
    if (current_conversation.size() >= 2) {
        for (size_t i = 0; i < current_conversation.size() - 1; i += 2) {
            if (i + 1 < current_conversation.size()) {
                training_queue.push(current_conversation[i], current_conversation[i + 1]);
                examples_added++;
            }
        }
    }
    
    file.close();
    std::cout << "[" << get_current_timestamp() << "] Added " << examples_added << " examples to training queue" << std::endl;
}

void create_minimal_dataset() {
    std::ofstream file("minimal_training.txt");
    if (file.is_open()) {
        file << "# Minimal training dataset\n";
        file << "Hello\tHi\n";
        file << "Hi\tHello\n";
        file << "Yes\tYes\n";
        file << "No\tNo\n";
        file.close();
        std::cout << "[" << get_current_timestamp() << "] Created minimal training dataset" << std::endl;
    }
}

// ============================================================================
// Validation Functions
// ============================================================================

bool validate_training_example(std::shared_ptr<lm::BPETokenizer> tokenizer, 
                              const std::string& input, const std::string& target) {
    try {
        auto input_tokens = tokenizer->encode(input);
        auto target_tokens = tokenizer->encode(target);
        
        if (input_tokens.empty() || target_tokens.empty()) {
            return false;
        }
        
        if (input_tokens.size() > 50 || target_tokens.size() > 50) {
            return false;
        }
        
        std::string decoded_input = tokenizer->decode(input_tokens);
        std::string decoded_target = tokenizer->decode(target_tokens);
        
        return (input == decoded_input && target == decoded_target);
    } catch (const std::exception& e) {
        return false;
    }
}

std::string validate_response(const std::string& response) {
    for (char c : response) {
        if (static_cast<unsigned char>(c) > 127 && static_cast<unsigned char>(c) < 192) {
            return "I'm not sure how to respond to that.";
        }
    }
    
    if (response.find("�") != std::string::npos) {
        return "I'm not sure how to respond to that.";
    }
    
    if (response.length() < 3) {
        return "I'm not sure how to respond to that.";
    }
    
    return response;
}

bool validate_model_state(std::shared_ptr<lm::ConversationModel> model) {
    try {
        std::string test_input = "Hello";
        std::string response = model->generate_response(test_input);
        
        if (response.empty() || response.find("�") != std::string::npos) {
            return false;
        }
        
        return true;
    } catch (const std::exception& e) {
        return false;
    }
}

bool needs_improvement(const std::string& response) {
    if (response.length() < 5) return true;
    
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

// ============================================================================
// Training Thread Function - AUTOMATIC TRAINING
// ============================================================================

void training_thread_func(
    std::shared_ptr<lm::ConversationModel> model,
    std::shared_ptr<lm::BPETokenizer> tokenizer,
    TrainingQueue& training_queue,
    const std::string& training_data_file,
    std::atomic<bool>& stop_training,
    std::atomic<int>& training_round,
    std::atomic<bool>& training_complete
) {
    SafeTrainingWrapper safe_wrapper(model, tokenizer);
    
    try {
        std::cout << "[" << get_current_timestamp() << "] Training thread started" << std::endl;
        
        while (!stop_training) {
            std::pair<std::string, std::string> example;
            int examples_processed = 0;
            
            // Process training examples from queue
            while (training_queue.pop(example) && examples_processed < 3 && !stop_training) {
                if (validate_training_example(tokenizer, example.first, example.second)) {
                    save_training_example(training_data_file, example.first, example.second);
                    examples_processed++;
                }
            }
            
            if (examples_processed > 0 && !stop_training) {
                auto dataset = load_training_dataset(training_data_file);
                
                if (dataset.size() > 0 && !stop_training) {
                    std::cout << "[" << get_current_timestamp() << "] Background training with " << dataset.size() << " examples" << std::endl;
                    
                    if (validate_model_state(model)) {
                        if (safe_wrapper.safe_train(dataset, 1, 0.00001f)) {
                            training_round++;
                            std::cout << "[" << get_current_timestamp() << "] Training completed. Round: " << training_round << std::endl;
                        } else {
                            std::cout << "[" << get_current_timestamp() << "] Trying alternative training" << std::endl;
                            if (safe_wrapper.alternative_train(dataset)) {
                                training_round++;
                                std::cout << "[" << get_current_timestamp() << "] Alternative training completed" << std::endl;
                            } else {
                                std::cerr << "[" << get_current_timestamp() << "] All training methods failed" << std::endl;
                            }
                        }
                    } else {
                        std::cerr << "[" << get_current_timestamp() << "] Model state invalid, skipping training" << std::endl;
                    }
                }
            }
            
            // Sleep with frequent stop checks
            for (int i = 0; i < 30 && !stop_training; i++) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[" << get_current_timestamp() << "] FATAL ERROR in training thread: " << e.what() << std::endl;
    }
    
    training_complete = true;
    std::cout << "[" << get_current_timestamp() << "] Training thread stopped" << std::endl;
}

// ============================================================================
// Command Processing
// ============================================================================

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
    
    std::string command;
    std::string argument;
    size_t space_pos = input.find(' ');
    
    if (space_pos != std::string::npos) {
        command = input.substr(1, space_pos - 1);
        argument = input.substr(space_pos + 1);
    } else {
        command = input.substr(1);
    }
    
    std::transform(command.begin(), command.end(), command.begin(), ::tolower);
    
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
        std::cout << "  /clear - Clear the conversation history" << std::endl;
        std::cout << "  /status - Show system status" << std::endl;
        std::cout << "  /trainfile <filename> - Load and train on conversations from a file" << std::endl;
        return true;
    }
    else if (command == "train") {
        std::cout << "Training round: " << training_round << std::endl;
        std::cout << "Pending training examples: " << training_queue.size() << std::endl;
        return true;
    }
    else if (command == "correct" && !current_conversation.empty()) {
        if (argument.empty()) {
            std::cout << "Usage: /correct <your better response>" << std::endl;
            return true;
        }
        
        if (current_conversation.size() >= 2) {
            std::string last_user_input = current_conversation[current_conversation.size() - 2];
            training_queue.push(last_user_input, argument);
            std::cout << "Correction saved. The AI will learn from this." << std::endl;
            
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
    else if (command == "trainfile") {
        if (argument.empty()) {
            std::cout << "Usage: /trainfile <filename>" << std::endl;
        } else {
            process_training_file(argument, training_queue);
        }
        return true;
    }
    else {
        std::cout << "Unknown command: " << command << std::endl;
        std::cout << "Type /help for available commands." << std::endl;
        return true;
    }
}

// ============================================================================
// Main Application with Automatic Training
// ============================================================================

int main() {
    std::cout << "[" << get_current_timestamp() << "] Starting interactive learning conversation system..." << std::endl;
    
    try {
        // Create minimal dataset
        create_minimal_dataset();
        
        // Initialize tokenizer
        std::cout << "[" << get_current_timestamp() << "] Creating BPE tokenizer..." << std::endl;
        auto tokenizer = std::make_shared<lm::BPETokenizer>();
        
        // Clean training data
        clean_training_data("training_data.txt");
        
        // Load training data
        auto training_dataset = load_training_dataset("cleaned_training_data.txt");
        
        if (training_dataset.size() == 0) {
            std::cerr << "[" << get_current_timestamp() << "] ERROR: No training data loaded!" << std::endl;
            return 1;
        }
        
        // Use subset for testing
        if (training_dataset.size() > 100) {
            lm::TrainingDataset small_dataset;
            for (size_t i = 0; i < std::min(training_dataset.size(), size_t(100)); i++) {
                small_dataset.add_example(training_dataset.examples()[i].input, 
                                         training_dataset.examples()[i].target);
            }
            training_dataset = small_dataset;
            std::cout << "[" << get_current_timestamp() << "] Using reduced dataset of " << training_dataset.size() << " examples" << std::endl;
        }
        
        std::cout << "[" << get_current_timestamp() << "] Loaded " << training_dataset.size() << " training examples." << std::endl;
        
        // Train tokenizer
        std::cout << "[" << get_current_timestamp() << "] Training tokenizer..." << std::endl;
        std::vector<std::string> tokenizer_training_data;
        std::vector<std::string> general_texts = {
            "Hello, how are you?", "I'm doing well, thank you!", 
            "What can I help you with today?", "The weather is nice today.",
            "I enjoy programming in C++.", "Machine learning is fascinating.",
            "Natural language processing enables computers to understand human language."
        };
        
        for (const auto& text : general_texts) {
            tokenizer_training_data.push_back(text);
        }
        
        for (const auto& example : training_dataset.examples()) {
            tokenizer_training_data.push_back(example.input);
            tokenizer_training_data.push_back(example.target);
        }
        
        tokenizer->train(tokenizer_training_data, 500);
        std::cout << "[" << get_current_timestamp() << "] Tokenizer training completed. Vocabulary size: " << tokenizer->vocab_size() << std::endl;
        
        // Initialize model
        std::cout << "[" << get_current_timestamp() << "] Initializing conversation model..." << std::endl;
        auto model = std::make_shared<lm::ConversationModel>(
            tokenizer->vocab_size(), 128, 2, 2, 512, 0.1f
        );
        model->set_tokenizer(tokenizer);
        model->set_verbose(false);
        model->set_max_response_length(50);
        
        // Try to load existing model
        bool model_loaded = false;
        std::vector<std::string> checkpoints = {"continuous_learning_model.bin", "final_model.bin"};
        
        for (const auto& checkpoint : checkpoints) {
            if (model->load_checkpoint(checkpoint)) {
                std::cout << "[" << get_current_timestamp() << "] Loaded model from checkpoint: " << checkpoint << std::endl;
                model_loaded = true;
                break;
            }
        }
        
        // Train if no checkpoint loaded
        if (!model_loaded) {
            std::cout << "[" << get_current_timestamp() << "] Training model with " << training_dataset.size() << " examples..." << std::endl;
            
            // Use safe training for initial training too
            SafeTrainingWrapper safe_wrapper(model, tokenizer);
            if (safe_wrapper.safe_train(training_dataset, 5, 0.005f)) {
                model->save_checkpoint("continuous_learning_model.bin");
                std::cout << "[" << get_current_timestamp() << "] Initial training completed and model saved." << std::endl;
            } else {
                std::cerr << "[" << get_current_timestamp() << "] Initial training failed!" << std::endl;
                // Try alternative training
                if (safe_wrapper.alternative_train(training_dataset)) {
                    model->save_checkpoint("continuous_learning_model.bin");
                    std::cout << "[" << get_current_timestamp() << "] Alternative training completed and model saved." << std::endl;
                } else {
                    std::cerr << "[" << get_current_timestamp() << "] All training methods failed for initial training!" << std::endl;
                    return 1;
                }
            }
        }
        
        // Validate model state
        if (!validate_model_state(model)) {
            std::cerr << "[" << get_current_timestamp() << "] Model is in an invalid state after loading/training!" << std::endl;
            return 1;
        }
        
        // Set up continuous learning system with automatic training
        TrainingQueue training_queue;
        std::atomic<bool> stop_training{false};
        std::atomic<bool> training_complete{false};
        std::atomic<int> training_round{0};
        std::atomic<bool> shutdown_requested{false};
        
        // Start automatic training thread
        std::cout << "[" << get_current_timestamp() << "] Starting automatic training thread..." << std::endl;
        std::thread training_thread(
            training_thread_func, 
            model, tokenizer, std::ref(training_queue), "training_data.txt",
            std::ref(stop_training), std::ref(training_round), std::ref(training_complete)
        );
        
        // Interactive conversation loop
        std::cout << "[" << get_current_timestamp() << "] Starting interactive conversation mode..." << std::endl;
        std::cout << "Type /help for available commands, /quit to exit." << std::endl;
        std::cout << "==========================================" << std::endl;
        
        std::vector<std::string> current_conversation;
        std::string input;
        int conversation_turns = 0;
        
        while (!shutdown_requested) {
            std::cout << "You: ";
            std::getline(std::cin, input);
            
            if (process_command(input, current_conversation, training_queue, training_round, 
                               shutdown_requested, stop_training, model, tokenizer)) {
                if (shutdown_requested) break;
                continue;
            }
            
            if (input.empty()) continue;
            
            current_conversation.push_back(input);
            conversation_turns++;
            
            // Generate response
            std::string response;
            try {
                response = model->generate_response(input);
                response = validate_response(response);
            } catch (const std::exception& e) {
                response = "I'm experiencing technical difficulties. Please try again.";
            }
            
            current_conversation.push_back(response);
            conversation_turns++;
            
            std::cout << "AI: " << response << std::endl;
            
            // Automatically add to training queue for continuous learning
            training_queue.push(input, response);
            
            // If response needs improvement, prompt user
            if (needs_improvement(response)) {
                std::cout << "[[ Would you like to provide a better response? Type '/correct <your better response>' or just continue. ]]" << std::endl;
            }
            
            // Limit conversation history
            if (current_conversation.size() > 20) {
                current_conversation.erase(current_conversation.begin(), current_conversation.begin() + 2);
            }
        }
        
        // Clean shutdown
        std::cout << "[" << get_current_timestamp() << "] Stopping training thread..." << std::endl;
        stop_training = true;
        training_queue.stop();
        
        if (training_thread.joinable()) {
            training_thread.join();
            std::cout << "[" << get_current_timestamp() << "] Training thread stopped" << std::endl;
        }
        
        // Save final model
        if (model->save_checkpoint("continuous_learning_model.bin")) {
            std::cout << "[" << get_current_timestamp() << "] Saved final model" << std::endl;
        }
        
        std::cout << "[" << get_current_timestamp() << "] System shutdown complete." << std::endl;
        std::cout << "Total training rounds completed: " << training_round << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "[" << get_current_timestamp() << "] Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

