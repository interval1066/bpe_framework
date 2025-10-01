#include <iostream>
#include "lm/config_manager.hpp"

int main() {
    lm::ConfigManager config("test_config.conf");
    
    // Test configuration access
    const auto& cfg = config.get_config();
    std::cout << "Vocabulary size: " << cfg.vocab_size << std::endl;
    std::cout << "Max sequence length: " << cfg.max_sequence_length << std::endl;
    
    // Test modification and saving
    config.get_config().vocab_size = 50000;
    config.save_config();
    
    return 0;
}

