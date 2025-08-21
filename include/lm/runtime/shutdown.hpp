#pragma once

#include <nlohmann/json.hpp>
#include <filesystem>
#include <chrono>

namespace lm::runtime {

class ShutdownHandler {
public:
    // Serialize state to JSON
    static void save_state(
        const std::filesystem::path& output_path,
        bool include_model_weights = false
    );
    
    // Cleanup hooks
    static void register_cleanup(void (*func)());
    static void execute_cleanup();
};

} // namespace lm::runtime
