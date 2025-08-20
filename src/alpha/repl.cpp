#include <iostream>
#include <string>
#include "lm/tokenizer/bpe_tokenizer.hpp"

void run_repl() {
    lm::BPETokenizer tokenizer;
    
    // Simple training for the alpha
    std::vector<std::string> corpus = {
        "hello world", "test input", "simple example"
    };
    tokenizer.train(corpus, 100);
    
    std::cout << "LM Framework Alpha\n> ";
    
    std::string input;
    while (std::getline(std::cin, input)) {
        if (input == "/exit") break;
        
        try {
            auto tokens = tokenizer.encode(input);
            std::cout << "Tokens: ";
            for (auto token : tokens) {
                std::cout << token << " ";
            }
            std::cout << "\n> ";
        } catch (const std::exception& e) {
            std::cout << "Error: " << e.what() << "\n> ";
        }
    }
    
    std::cout << "Saving session...\n";
    tokenizer.save("alpha_session.bpe");
}

int main() {
    try {
        run_repl();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
