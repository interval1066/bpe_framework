#include "lm/runtime/shutdown.hpp"
#include "lm/runtime/init.hpp"
#include <iomanip>
#include <ctime>

namespace lm::runtime {

// Helper function to format timestamp
std::string format_timestamp(int64_t timestamp_ns) {
    std::time_t time = timestamp_ns / 1000000000;
    std::tm* tm = std::localtime(&time);
    
    if (tm) {
        std::ostringstream oss;
        oss << std::put_time(tm, "%Y-%m-%d %H:%M:%S");
        return oss.str();
    }
    return "invalid_timestamp";
}

// Generate a comprehensive state report
std::string generate_state_report(const nlohmann::json& state) {
    std::ostringstream report;
    
    report << "=== LM Framework State Report ===\n\n";
    
    // Basic information
    if (state.contains("metadata")) {
        const auto& metadata = state["metadata"];
        report << "Shutdown Time: ";
        if (metadata.contains("shutdown_time")) {
            report << format_timestamp(metadata["shutdown_time"].get<int64_t>());
        } else {
            report << "unknown";
        }
        report << "\nVersion: " << metadata.value("version", "unknown") << "\n\n";
    }
    
    // Tokenizer state
    if (state.contains("tokenizer")) {
        const auto& tokenizer = state["tokenizer"];
        report << "Tokenizer:\n";
        report << "  Initialized: " << tokenizer.value("runtime/initialized", false) << "\n";
        
        if (tokenizer.contains("type")) {
            report << "  Type: " << tokenizer["type"] << "\n";
        }
        if (tokenizer.contains("vocab_size")) {
            report << "  Vocab Size: " << tokenizer["vocab_size"] << "\n";
        }
        report << "\n";
    }
    
    // Model state
    if (state.contains("model")) {
        const auto& model = state["model"];
        report << "Model:\n";
        report << "  Loaded: " << model.value("runtime/loaded", false) << "\n";
        
        if (model.contains("layers")) {
            report << "  Layers: " << model["layers"] << "\n";
        }
        if (model.contains("dim")) {
            report << "  Dimension: " << model["dim"] << "\n";
        }
        report << "\n";
    }
    
    // Threading state
    if (state.contains("threading")) {
        const auto& threading = state["threading"];
        report << "Threading:\n";
        report << "  Active Threads: " << threading.value("active_threads", 0) << "\n";
        report << "  Queued Tasks: " << threading.value("queued_tasks", 0) << "\n";
        report << "\n";
    }
    
    return report.str();
}

} // namespace lm::runtime
