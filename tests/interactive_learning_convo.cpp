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

// Helper function to get current timestamp
std::string get_current_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %X");
    return ss.str();
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

// Background training thread function
void training_thread_func(
    std::shared_ptr<lm::ConversationModel> model,
    std::shared_ptr<lm::BPETokenizer> tokenizer,
    TrainingQueue& training_queue,
    const std::string& training_data_file,
    std::atomic<bool>& stop_training,
    std::atomic<int>& training_round,
    std::atomic<bool>& training_complete
) {
    while (!stop_training) {
        // Check for new training examples
        std::pair<std::string, std::string> example;
        int examples_processed = 0;
        
        while (training_queue.pop(example) && examples_processed < 10) {
            // Save the example to the training data file
            save_training_example(training_data_file, example.first, example.second);
            examples_processed++;
        }
        
        if (examples_processed > 0) {
            // Reload the training dataset
            auto dataset = load_training_dataset(training_data_file);
            
            if (dataset.size() > 0) {
                std::cout << "[" << get_current_timestamp() << "] Background training with " << dataset.size() << " examples..." << std::endl;
                
                // Fine-tune the model with the updated dataset
                try {
                    // Shuffle the dataset
                    dataset.shuffle();
                    
                    // Train with a lower learning rate for fine-tuning
                    model->train(dataset, 3, 0.001f); // 3 epochs with reduced learning rate
                    
                    training_round++;
                    std::cout << "[" << get_current_timestamp() << "] Background training completed. Training round: " << training_round << std::endl;
                    
                    // Save model checkpoint periodically
                    if (training_round % 5 == 0) {
                        std::string checkpoint_name = "continuous_learning_checkpoint_round_" + std::to_string(training_round) + ".bin";
                        if (model->save_checkpoint(checkpoint_name)) {
                            std::cout << "[" << get_current_timestamp() << "] Saved checkpoint: " << checkpoint_name << std::endl;
                        }
                    }
                } catch (const std::exception& e) {
                    std::cerr << "[" << get_current_timestamp() << "] Error during background training: " << e.what() << std::endl;
                }
            }
        }
        
        // Sleep for a while before checking again
        std::this_thread::sleep_for(std::chrono::seconds(30));
    }
    
    training_complete = true;
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
    std::atomic<bool>& shutdown_requested
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
    std::atomic<bool>& shutdown_requested
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
        if (process_command(input, current_conversation, training_queue, training_round, shutdown_requested)) {
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
        std::string response = model->generate_response(input);
        
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
        // Initialize tokenizer
        std::cout << "[" << get_current_timestamp() << "] Creating BPE tokenizer..." << std::endl;
        auto tokenizer = std::make_shared<lm::BPETokenizer>();
        
        // Load training data from file
        std::cout << "[" << get_current_timestamp() << "] Loading training data from file..." << std::endl;
        auto training_dataset = load_training_dataset("training_data.txt");
        
        if (training_dataset.size() == 0) {
            std::cerr << "[" << get_current_timestamp() << "] ERROR: No training data loaded!" << std::endl;
            return 1;
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
        tokenizer->train(tokenizer_training_data, 1000);
        std::cout << "[" << get_current_timestamp() << "] Tokenizer training completed. Vocabulary size: " 
                  << tokenizer->vocab_size() << std::endl;
        
        // Initialize conversation model with explicit parameters
        std::cout << "[" << get_current_timestamp() << "] Initializing conversation model..." << std::endl;
        auto model = std::make_shared<lm::ConversationModel>(
            tokenizer->vocab_size(), // vocab_size
            512,                     // d_model
            6,                       // n_layers
            8,                       // n_heads
            2048,                    // d_ff
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
            
            // Train for multiple epochs with learning rate decay
            const size_t num_epochs = 30;
            float learning_rate = 0.01f;
            
            for (size_t epoch = 0; epoch < num_epochs; epoch++) {
                std::cout << "[" << get_current_timestamp() << "] Epoch " << (epoch + 1) 
                          << "/" << num_epochs << " (LR: " << learning_rate << ")" << std::endl;
                
                // Shuffle the dataset for each epoch
                training_dataset.shuffle();
                
                // Train on the shuffled dataset
                model->train(training_dataset, 1, learning_rate);
                
                // Decay learning rate
                learning_rate *= 0.95f;
            }
            
            // Save initial model
            if (model->save_checkpoint("continuous_learning_model.bin")) {
                std::cout << "[" << get_current_timestamp() << "] Saved initial model." << std::endl;
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
        interactive_conversation(model, training_queue, training_round, shutdown_requested);
        
        // Stop the training thread
        stop_training = true;
        training_queue.stop();
        if (training_thread.joinable()) {
            training_thread.join();
        }
        
        // Save the final model
        if (model->save_checkpoint("continuous_learning_model.bin")) {
            std::cout << "[" << get_current_timestamp() << "] Saved final model with continuous learning." << std::endl;
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
