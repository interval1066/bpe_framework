// src/chat_demo.cpp
#include <lm/inference/inference_engine.hpp>
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <model_path> <tokenizer_path> [sampler_type] [temperature]" << std::endl;
        std::cout << "Sampler types: greedy, temperature, top_k, top_p" << std::endl;
        std::cout << "Temperature range: 0.1 to 2.0 (default: 0.8)" << std::endl;
        return 1;
    }
    
    std::string model_path = argv[1];
    std::string tokenizer_path = argv[2];
    std::string sampler_type = "top_p"; // Default
    float temperature = 0.8f; // Default
    
    if (argc >= 4) {
        sampler_type = argv[3];
    }
    
    if (argc >= 5) {
        try {
            temperature = std::stof(argv[4]);
            if (temperature < 0.1f || temperature > 2.0f) {
                std::cout << "Temperature should be between 0.1 and 2.0. Using default 0.8" << std::endl;
                temperature = 0.8f;
            }
        } catch (const std::exception& e) {
            std::cout << "Invalid temperature value. Using default 0.8" << std::endl;
            temperature = 0.8f;
        }
    }
    
    try {
        // Initialize the inference engine
        lm::InferenceEngine engine(model_path, tokenizer_path, sampler_type, temperature);
        
        std::cout << "=== Chat Demo ===" << std::endl;
        std::cout << "Model: " << model_path << std::endl;
        std::cout << "Tokenizer: " << tokenizer_path << std::endl;
        std::cout << "Sampler: " << sampler_type << std::endl;
        std::cout << "Temperature: " << temperature << std::endl;
        std::cout << "==============================================" << std::endl;
        std::cout << "Type 'quit' to exit, 'clear' to reset conversation, 'save <file>' to save conversation" << std::endl;
        std::cout << "Type 'load <file>' to load a conversation, 'help' to show this message" << std::endl;
        std::cout << "==============================================" << std::endl;
        
        std::string input;
        while (true) {
            std::cout << "\nYou: ";
            std::getline(std::cin, input);
            
            if (input == "quit" || input == "exit") {
                break;
            } else if (input == "clear") {
                engine.reset_conversation();
                std::cout << "Conversation history cleared." << std::endl;
                continue;
            } else if (input.find("save ") == 0) {
                std::string filename = input.substr(5);
                if (!filename.empty()) {
                    engine.save_conversation(filename);
                    std::cout << "Conversation saved to: " << filename << std::endl;
                } else {
                    std::cout << "Please specify a filename: save <filename>" << std::endl;
                }
                continue;
            } else if (input.find("load ") == 0) {
                std::string filename = input.substr(5);
                if (!filename.empty()) {
                    try {
                        engine.load_conversation(filename);
                        std::cout << "Conversation loaded from: " << filename << std::endl;
                    } catch (const std::exception& e) {
                        std::cerr << "Error loading conversation: " << e.what() << std::endl;
                    }
                } else {
                    std::cout << "Please specify a filename: load <filename>" << std::endl;
                }
                continue;
            } else if (input == "help") {
                std::cout << "Available commands:" << std::endl;
                std::cout << "  quit/exit - Exit the program" << std::endl;
                std::cout << "  clear - Reset the conversation" << std::endl;
                std::cout << "  save <file> - Save conversation to file" << std::endl;
                std::cout << "  load <file> - Load conversation from file" << std::endl;
                std::cout << "  help - Show this help message" << std::endl;
                continue;
            } else if (input.empty()) {
                continue;
            }
            
            try {
                std::cout << "AI: ";
                std::string response = engine.chat(input);
                std::cout << response << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Error generating response: " << e.what() << std::endl;
            }
        }
        
        std::cout << "Goodbye!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize inference engine: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
