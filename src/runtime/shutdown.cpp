#include "lm/runtime/shutdown.hpp"
#include "lm/runtime/init.hpp"
#include "lm/tokenizer/bpe_tokenizer.hpp"
#include <fstream>
#include <vector>
#include <mutex>
#include <sstream>
#include <iostream>

namespace lm::runtime {

namespace {
    std::vector<void (*)()> cleanup_functions;
    std::mutex cleanup_mutex;
}

// Serialize tokenizer state to JSON
nlohmann::json serialize_tokenizer_state() {
    auto& system_state = SystemState::get_instance();
    nlohmann::json tokenizer_state;
    
    // Get tokenizer configuration from system state
    try {
        const auto& config = system_state.config();
        if (config.contains("tokenizer")) {
            tokenizer_state = config["tokenizer"];
        }
        
        // Add runtime information
        tokenizer_state["runtime"] = {
            {"initialized", system_state.is_tokenizer_ready()},
            {"timestamp", std::chrono::system_clock::now().time_since_epoch().count()}
        };
        
    } catch (const std::exception& e) {
        tokenizer_state["error"] = std::string("Failed to serialize tokenizer state: ") + e.what();
    }
    
    return tokenizer_state;
}

// Serialize model state to JSON
nlohmann::json serialize_model_state(bool include_weights) {
    auto& system_state = SystemState::get_instance();
    nlohmann::json model_state;
    
    try {
        const auto& config = system_state.config();
        if (config.contains("model")) {
            model_state = config["model"];
        }
        
        // Add runtime information
        model_state["runtime"] = {
            {"loaded", system_state.is_model_loaded()},
            {"timestamp", std::chrono::system_clock::now().time_since_epoch().count()}
        };
        
        if (include_weights) {
            // Placeholder for actual weight serialization
            model_state["weights"] = {
                {"serialized", false},
                {"message", "Weight serialization not yet implemented"}
            };
        }
        
    } catch (const std::exception& e) {
        model_state["error"] = std::string("Failed to serialize model state: ") + e.what();
    }
    
    return model_state;
}

// Serialize threading state to JSON
nlohmann::json serialize_thread_pool_stats() {
    nlohmann::json threading_state;
    
    try {
        // Placeholder for actual thread pool statistics
        // This would normally come from ThreadPool::get_stats()
        threading_state = {
            {"active_threads", 0},
            {"queued_tasks", 0},
            {"completed_tasks", 0},
            {"thread_pool_initialized", false}
        };
        
    } catch (const std::exception& e) {
        threading_state["error"] = std::string("Failed to serialize threading state: ") + e.what();
    }
    
    return threading_state;
}

void ShutdownHandler::save_state(
    const std::filesystem::path& output_path,
    bool include_model_weights) 
{
    try {
        nlohmann::json state;
        
        // Capture framework state
        auto& system_state = SystemState::get_instance();
        
        // Add system configuration
        state["config"] = system_state.config();
        
        // Add component states
        state["tokenizer"] = serialize_tokenizer_state();
        state["model"] = serialize_model_state(include_model_weights);
        state["threading"] = serialize_thread_pool_stats();
        
        // Add shutdown metadata
        state["metadata"] = {
            {"shutdown_time", std::chrono::system_clock::now().time_since_epoch().count()},
            {"include_weights", include_model_weights},
            {"version", "0.1.0"},
            {"format_version", 1}
        };
        
        // Write to file
        std::ofstream file(output_path);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file for writing: " + output_path.string());
        }
        
        file << state.dump(2); // Pretty print with 2-space indentation
        file.close();
        
        std::cout << "Framework state saved to: " << output_path << std::endl;
        
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to save state: " + std::string(e.what()));
    }
}

void ShutdownHandler::register_cleanup(void (*func)()) {
    std::lock_guard<std::mutex> lock(cleanup_mutex);
    cleanup_functions.push_back(func);
}

void ShutdownHandler::execute_cleanup() {
    std::lock_guard<std::mutex> lock(cleanup_mutex);
    
    // Execute cleanup functions in reverse order (LIFO)
    for (auto it = cleanup_functions.rbegin(); it != cleanup_functions.rend(); ++it) {
        try {
            (*it)();
        } catch (const std::exception& e) {
            // Log error but continue with other cleanup functions
            std::cerr << "Cleanup function error: " << e.what() << std::endl;
        }
    }
    
    cleanup_functions.clear();
}

} // namespace lm::runtime

