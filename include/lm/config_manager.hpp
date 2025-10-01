#pragma once

#include <string>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>

namespace lm {

struct TrainingConfig {
    // Stability and training parameters
    double stability_factor = 0.3;
    bool aggressive_normalization = true;
    size_t min_sequence_length = 1;
    size_t max_sequence_length = 2000;
    
    // Literary text handling
    bool preserve_paragraphs = true;
    bool preserve_punctuation = true;
    bool handle_contractions = true;
    
    // BPE training parameters
    size_t vocab_size = 30000;
    size_t min_frequency = 2;
    int max_iterations = 10000;
    
    // Debug and logging
    bool debug_logging = false;
    bool verbose_output = true;
    
    // Display all parameters
    void print() const {
        std::cout << "=== Training Configuration ===" << std::endl;
        std::cout << "Stability Factor: " << stability_factor << std::endl;
        std::cout << "Aggressive Normalization: " << (aggressive_normalization ? "true" : "false") << std::endl;
        std::cout << "Min Sequence Length: " << min_sequence_length << std::endl;
        std::cout << "Max Sequence Length: " << max_sequence_length << std::endl;
        std::cout << "Preserve Paragraphs: " << (preserve_paragraphs ? "true" : "false") << std::endl;
        std::cout << "Preserve Punctuation: " << (preserve_punctuation ? "true" : "false") << std::endl;
        std::cout << "Handle Contractions: " << (handle_contractions ? "true" : "false") << std::endl;
        std::cout << "Vocabulary Size: " << vocab_size << std::endl;
        std::cout << "Min Frequency: " << min_frequency << std::endl;
        std::cout << "Max Iterations: " << max_iterations << std::endl;
        std::cout << "Debug Logging: " << (debug_logging ? "true" : "false") << std::endl;
        std::cout << "Verbose Output: " << (verbose_output ? "true" : "false") << std::endl;
        std::cout << "==============================" << std::endl;
    }
};

class ConfigManager {
public:
    ConfigManager(const std::string& config_path = "bpe_training.conf");
    
    // Main configuration methods
    bool load_config();
    bool save_config() const;
    void create_default_config();
    
    // Access to configuration
    const TrainingConfig& get_config() const { return config_; }
    TrainingConfig& get_config() { return config_; }
    
    // Individual parameter accessors
    double get_stability_factor() const { return config_.stability_factor; }
    size_t get_vocab_size() const { return config_.vocab_size; }
    size_t get_max_sequence_length() const { return config_.max_sequence_length; }
    bool get_debug_logging() const { return config_.debug_logging; }
    
    // Individual parameter setters
    void set_stability_factor(double factor) { config_.stability_factor = factor; }
    void set_vocab_size(size_t size) { config_.vocab_size = size; }
    void set_max_sequence_length(size_t length) { config_.max_sequence_length = length; }
    void set_debug_logging(bool debug) { config_.debug_logging = debug; }
    
    // Validation
    bool validate_config() const;
    
    // File operations
    std::string get_config_path() const { return config_path_; }
    void set_config_path(const std::string& path) { config_path_ = path; }
    
    // Utility methods
    void print_config() const { config_.print(); }
    void reset_to_defaults();

private:
    std::string config_path_;
    TrainingConfig config_;
    
    // Parsing helpers
    bool parse_line(const std::string& line, std::string& key, std::string& value);
    void trim(std::string& str);
    bool parse_bool(const std::string& str) const;
    double parse_double(const std::string& str) const;
    int parse_int(const std::string& str) const;
    size_t parse_size_t(const std::string& str) const;
    
    // Writing helpers
    std::string bool_to_str(bool value) const { return value ? "true" : "false"; }
};

} // namespace lm
