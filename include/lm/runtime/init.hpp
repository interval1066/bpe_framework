// Runtime Initialization Header File

//Here's the complete `include/lm/runtime/init.hpp` file:

//```cpp
#pragma once

#include <string>
#include <nlohmann/json.hpp>
#include <filesystem>

namespace lm::runtime {

class SystemState {
public:
    // Singleton access
    static SystemState& get_instance();
    
    // Initialize from JSON config
    void initialize(const std::filesystem::path& config_path);
    
    // Configuration accessors
    const nlohmann::json& config() const noexcept;
    std::string get_string(const std::string& key) const;
    int get_int(const std::string& key, int default_val = 0) const;
    
    // Subsystem states
    bool is_tokenizer_ready() const noexcept;
    bool is_model_loaded() const noexcept;

private:
    SystemState() = default; // Private constructor
    nlohmann::json config_;
    bool tokenizer_ready_ = false;
    bool model_loaded_ = false;
};

} // namespace lm::runtime
/*```

This header provides the interface for the framework initialization system with:

1. **Singleton pattern** for global system state access
2. **JSON configuration** loading and access methods
3. **Subsystem state tracking** for tokenizer and model
4. **Type-safe configuration access** with default values

The implementation (in the corresponding `.cpp` file) handles:
- JSON configuration parsing and validation
- Subsystem initialization sequencing
- Error handling for malformed configurations
- State management across the framework

This initialization system provides a centralized way to configure and manage the LM framework components.*/