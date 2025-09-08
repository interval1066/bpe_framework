// main.cpp
#include "lm/model/conversation_model.hpp"
#include "lm/tokenizer/bpe_tokenizer.hpp"
#include <iostream>

int main() {
    // Initialize tokenizer
    auto tokenizer = std::make_shared<lm::BPETokenizer>();
    
    // Train or load tokenizer
    std::vector<std::string> training_data = {
        "Hello, how are you?",
        "I'm doing well, thank you!",
        "What can I help you with today?",
        // ... more training data
    };
    tokenizer->train(training_data, 10000);
    
    // Initialize conversation model
    lm::ConversationModel model(tokenizer->vocab_size());
    model.set_tokenizer(tokenizer);
    
    // Train the model
    std::vector<std::string> conversations = {
        "<|user|>Hello<|endoftext|><|assistant|>Hi there! How can I help you?<|endoftext|>",
        "<|user|>What's the weather like?<|endoftext|><|assistant|>I'm not sure, I don't have access to real-time weather data.<|endoftext|>",
        // ... more conversation examples
    };
    model.train(conversations);
    
    // Generate responses
    std::string response = model.generate_response("Hello, how are you?");
    std::cout << "Response: " << response << std::endl;
    
    // Save the model
    model.save_model("conversation_model.bin");
    
    return 0;
}
