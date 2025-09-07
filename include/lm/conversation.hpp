// include/lm/conversation.hpp
#pragma once

#include <string>
#include <vector>
#include <map>
#include <chrono>
#include <memory>
#include <cereal/types/vector.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/chrono.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/archives/binary.hpp>

namespace lm {

// Enum for different speaker types
enum class SpeakerType {
    USER,
    ASSISTANT,
    SYSTEM,
    UNKNOWN
};

// Convert SpeakerType to string
inline std::string speaker_type_to_string(SpeakerType type) {
    switch (type) {
        case SpeakerType::USER: return "user";
        case SpeakerType::ASSISTANT: return "assistant";
        case SpeakerType::SYSTEM: return "system";
        default: return "unknown";
    }
}

// Convert string to SpeakerType
inline SpeakerType string_to_speaker_type(const std::string& str) {
    if (str == "user") return SpeakerType::USER;
    if (str == "assistant") return SpeakerType::ASSISTANT;
    if (str == "system") return SpeakerType::SYSTEM;
    return SpeakerType::UNKNOWN;
}

// Represents a single turn in a conversation
struct ConversationTurn {
    SpeakerType speaker;
    std::string text;
    std::vector<int> tokens;  // Tokenized representation
    std::chrono::system_clock::time_point timestamp;
    std::map<std::string, std::string> metadata;  // Additional metadata
    
    ConversationTurn(SpeakerType speaker_type = SpeakerType::UNKNOWN, 
                    const std::string& text = "",
                    const std::map<std::string, std::string>& metadata = {})
        : speaker(speaker_type), text(text), metadata(metadata) {
        timestamp = std::chrono::system_clock::now();
    }
    
    // Cereal serialization
    template <class Archive>
    void serialize(Archive& archive) {
        archive(
            cereal::make_nvp("speaker", reinterpret_cast<int&>(speaker)),
            cereal::make_nvp("text", text),
            cereal::make_nvp("tokens", tokens),
            cereal::make_nvp("timestamp", timestamp),
            cereal::make_nvp("metadata", metadata)
        );
    }
};

// Represents a complete conversation with multiple turns
struct Conversation {
    std::vector<ConversationTurn> turns;
    std::string domain;  // e.g., "customer_service", "general_chat", "technical_support"
    std::string language;
    std::map<std::string, std::string> metadata;
    std::chrono::system_clock::time_point start_time;
    std::chrono::system_clock::time_point end_time;
    
    Conversation(const std::string& domain = "general_chat",
                const std::string& language = "en",
                const std::map<std::string, std::string>& metadata = {})
        : domain(domain), language(language), metadata(metadata) {
        start_time = std::chrono::system_clock::now();
    }
    
    // Add a turn to the conversation
    void add_turn(SpeakerType speaker, const std::string& text,
                 const std::map<std::string, std::string>& metadata = {}) {
        turns.emplace_back(speaker, text, metadata);
        end_time = std::chrono::system_clock::now();
    }
    
    // Get the last turn
    ConversationTurn& last_turn() {
        if (turns.empty()) {
            throw std::out_of_range("No turns in conversation");
        }
        return turns.back();
    }
    
    // Get the number of turns
    size_t size() const {
        return turns.size();
    }
    
    // Check if conversation is empty
    bool empty() const {
        return turns.empty();
    }
    
    // Clear all turns
    void clear() {
        turns.clear();
        start_time = std::chrono::system_clock::now();
    }
    
    // Get conversation duration in seconds
    double duration() const {
        if (turns.empty()) return 0.0;
        auto duration = end_time - start_time;
        return std::chrono::duration<double>(duration).count();
    }
    
    // Cereal serialization
    template <class Archive>
    void serialize(Archive& archive) {
        archive(
            cereal::make_nvp("turns", turns),
            cereal::make_nvp("domain", domain),
            cereal::make_nvp("language", language),
            cereal::make_nvp("metadata", metadata),
            cereal::make_nvp("start_time", start_time),
            cereal::make_nvp("end_time", end_time)
        );
    }
};

// Helper functions for conversation processing
namespace conversation_utils {

// Extract text from a range of turns
inline std::string extract_text(const std::vector<ConversationTurn>& turns,
                               size_t start_idx = 0, size_t end_idx = 0) {
    if (end_idx == 0) end_idx = turns.size();
    if (start_idx >= end_idx || end_idx > turns.size()) return "";
    
    std::string result;
    for (size_t i = start_idx; i < end_idx; i++) {
        result += speaker_type_to_string(turns[i].speaker) + ": " + turns[i].text + "\n";
    }
    return result;
}

// Create a training pair from conversation turns
inline std::pair<std::string, std::string> create_training_pair(
    const std::vector<ConversationTurn>& turns, size_t context_length) {
    
    if (turns.size() < 2) return {"", ""};
    
    // Use the last 'context_length' turns as context (excluding the last turn)
    size_t start_idx = turns.size() > context_length + 1 ? 
                      turns.size() - context_length - 1 : 0;
    size_t end_idx = turns.size() - 1;
    
    std::string context = extract_text(turns, start_idx, end_idx);
    std::string target = turns.back().text;
    
    return {context, target};
}

// Calculate turns-based context window
inline std::vector<ConversationTurn> get_context_window(
    const std::vector<ConversationTurn>& turns, size_t max_turns) {
    
    if (turns.size() <= max_turns) return turns;
    
    return std::vector<ConversationTurn>(
        turns.end() - max_turns, turns.end());
}

} // namespace conversation_utils

} // namespace lm
