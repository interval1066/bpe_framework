#include "lm/runtime/init.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <stdexcept>

nlohmann::json load_config(const std::string& path) {
    try {
        std::ifstream file(path);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open config file: " + path);
        }
        
        nlohmann::json config;
        file >> config;
        return config;
        
    } catch (const std::exception& e) {
        // Fallback to default config if file doesn't exist or is invalid
        return nlohmann::json{
            {"alpha", {
                {"prompt", "> "},
                {"save_on_exit", true}
            }},
            {"tokenizer", {
                {"type", "bpe"},
                {"vocab_size", 100},
                {"dummy_data", true}
            }},
            {"model", {
                {"layers", 2},
                {"dim", 64}
            }}
        };
    }
}

void save_config(const nlohmann::json& config, const std::string& path) {
    try {
        std::ofstream file(path);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file for writing: " + path);
        }
        
        file << config.dump(2); // Pretty print with 2-space indentation
        
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to save config: " + std::string(e.what()));
    }
}
