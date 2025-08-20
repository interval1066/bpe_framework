#include "lm/runtime/init.hpp"
#include "lm/runtime/shutdown.hpp"
#include <sstream>
#include <fstream>
#include <vector>
#include <mutex>

namespace lm::runtime {

namespace {
    std::vector<void (*)()> cleanup_functions;
    std::mutex cleanup_mutex;
}

void ShutdownHandler::save_state(
    const std::filesystem::path& output_path) 
{
    try {
        nlohmann::json state;
        
        // Capture framework state
        auto& system_state = SystemState::get_instance();
        state["config"] = system_state.config();
        
        // TODO: Add tokenizer state
        // state["tokenizer"] = serialize_tokenizer_state();
        
        // TODO: Add model state (optionally with weights)
        // state["model"] = serialize_model_state(include_model_weights);
        
        // TODO: Add threading state
        // state["threading"] = serialize_thread_pool_stats();
        
        // Write to file
        std::ofstream file(output_path);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file for writing: " + output_path.string());
        }
        
        file << state.dump(2); // Pretty print with 2-space indentation
        file.close();
        
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
            // TODO: Add proper logging
        }
    }
    
    cleanup_functions.clear();
}

} // namespace lm::runtime

