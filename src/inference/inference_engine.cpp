// src/inference/inference_engine.cpp
#include <lm/inference/inference_engine.hpp>
#include <lm/models/transformer_model.hpp> // Ensure this is included
#include <lm/generation/sampler.hpp>
#include <lm/generation/random_sampler.hpp>
#include <lm/generation/greedy_sampler.hpp>
#include <lm/generation/temperature_sampler.hpp>
#include <lm/generation/topk_sampler.hpp>
#include <lm/generation/topp_sampler.hpp>
#include <fstream>
#include <stdexcept>
#include <memory>

namespace lm {

InferenceEngine::InferenceEngine(const std::string& model_path, 
                               const std::string& tokenizer_path,
                               const std::string& sampler_type,
                               float temperature) {
    load_model(model_path);
    load_tokenizer(tokenizer_path);
    
    // Create the specified sampler
    auto sampler = create_sampler(sampler_type, temperature);
    
    // Create conversation manager
    conversation_manager_ = std::make_unique<ConversationManager>(
        model_, tokenizer_, std::move(sampler)
    );
}

std::string InferenceEngine::chat(const std::string& input, 
                                size_t max_length,
                                float temperature) {
    if (!conversation_manager_) {
        throw std::runtime_error("Conversation manager not initialized");
    }
    
    // Update sampler temperature if needed
    if (temperature != 0.8f) { // Default temperature is 0.8
        conversation_manager_->set_sampler(create_sampler("top_p", temperature));
    }
    
    return conversation_manager_->generate_response(input, max_length, temperature);
}

void InferenceEngine::reset_conversation() {
    if (conversation_manager_) {
        conversation_manager_->clear_history();
    }
}

void InferenceEngine::save_conversation(const std::string& path) const {
    std::ofstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for saving conversation: " + path);
    }
    
    auto history = conversation_manager_->get_history();
    for (const auto& message : history) {
        file << message << "\n";
    }
}

void InferenceEngine::load_conversation(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open conversation file: " + path);
    }
    
    reset_conversation();
    
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            // Determine if it's a user or AI message
            bool is_user = line.find("User: ") == 0;
            std::string content = is_user ? line.substr(6) : line.substr(4);
            
            if (conversation_manager_) {
                // We need to add a method to ConversationManager to add pre-existing messages
                // For now, we'll just store them in history
                conversation_manager_->add_to_history(content, is_user);
            }
        }
    }
}

void InferenceEngine::load_model(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) {
        throw std::runtime_error("Failed to open model file: " + path);
    }
    
    model_ = std::make_shared<TransformerModel>();
    model_->deserialize(ifs);
}

void InferenceEngine::load_tokenizer(const std::string& path) {
    tokenizer_ = std::make_shared<BPETokenizer>();
    tokenizer_->load(path);
}

std::unique_ptr<Sampler> InferenceEngine::create_sampler(const std::string& type, float temperature) {
    if (type == "greedy") {
        return std::make_unique<GreedySampler>();
    } else if (type == "temperature") {
        return std::make_unique<TemperatureSampler>(temperature);
    } else if (type == "top_k") {
        return std::make_unique<TopKSampler>(50, temperature); // Default k=50
    } else if (type == "top_p") {
        return std::make_unique<TopPSampler>(0.9f, temperature); // Default p=0.9
    } else if (type == "random") {
        return std::make_unique<RandomSampler>(temperature); // Pass temperature to RandomSampler
    } else {
        throw std::invalid_argument("Unknown sampler type: " + type);
    }
}

} // namespace lm

