// src/conversation_manager.cpp
#include "lm/conversation_manager.hpp"
#include <random>
#include <algorithm>
#include <fstream>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/chrono.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/archives/binary.hpp>

namespace lm {

ConversationManager::ConversationManager() {}

ConversationManager::~ConversationManager() {}

std::string ConversationManager::generate_id() const {
    static const char alphanum[] =
        "0123456789"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz";
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, sizeof(alphanum) - 2);
    
    std::string id;
    for (int i = 0; i < 16; ++i) {
        id += alphanum[dis(gen)];
    }
    
    return id;
}

std::string ConversationManager::create_conversation(const std::string& title) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::string id = generate_id();
    auto conversation = std::make_shared<Conversation>();
    
    if (!title.empty()) {
        conversation->metadata["title"] = title;
    }
    
    conversations_[id] = conversation;
    return id;
}

std::shared_ptr<Conversation> ConversationManager::get_conversation(const std::string& id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = conversations_.find(id);
    if (it != conversations_.end()) {
        return it->second;
    }
    
    return nullptr;
}

std::vector<std::string> ConversationManager::list_conversations() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<std::string> ids;
    for (const auto& pair : conversations_) {
        ids.push_back(pair.first);
    }
    
    return ids;
}

void ConversationManager::add_message(const std::string& conversation_id, 
                                     const std::string& role, 
                                     const std::string& content) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = conversations_.find(conversation_id);
    if (it == conversations_.end()) {
        throw std::runtime_error("Conversation not found: " + conversation_id);
    }
    
    SpeakerType speaker_type = string_to_speaker_type(role);
    it->second->add_turn(speaker_type, content);
}

std::vector<ConversationTurn> ConversationManager::get_history(const std::string& conversation_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = conversations_.find(conversation_id);
    if (it == conversations_.end()) {
        throw std::runtime_error("Conversation not found: " + conversation_id);
    }
    
    return it->second->turns;
}

bool ConversationManager::save_conversations(const std::string& path) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    try {
        std::ofstream ofs(path, std::ios::binary);
        cereal::BinaryOutputArchive archive(ofs);
        archive(conversations_);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error saving conversations: " << e.what() << std::endl;
        return false;
    }
}

bool ConversationManager::load_conversations(const std::string& path) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    try {
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs.is_open()) {
            std::cerr << "Could not open file: " << path << std::endl;
            return false;
        }
        
        cereal::BinaryInputArchive archive(ifs);
        archive(conversations_);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading conversations: " << e.what() << std::endl;
        return false;
    }
}

bool ConversationManager::delete_conversation(const std::string& id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    return conversations_.erase(id) > 0;
}

void ConversationManager::set_title(const std::string& conversation_id, const std::string& title) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = conversations_.find(conversation_id);
    if (it == conversations_.end()) {
        throw std::runtime_error("Conversation not found: " + conversation_id);
    }
    
    it->second->metadata["title"] = title;
}

std::string ConversationManager::get_title(const std::string& conversation_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = conversations_.find(conversation_id);
    if (it == conversations_.end()) {
        throw std::runtime_error("Conversation not found: " + conversation_id);
    }
    
    auto title_it = it->second->metadata.find("title");
    if (title_it != it->second->metadata.end()) {
        return title_it->second;
    }
    
    return "Untitled Conversation";
}

std::map<std::string, std::string> ConversationManager::get_metadata(const std::string& conversation_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = conversations_.find(conversation_id);
    if (it == conversations_.end()) {
        throw std::runtime_error("Conversation not found: " + conversation_id);
    }
    
    return it->second->metadata;
}

void ConversationManager::update_metadata(const std::string& conversation_id, 
                                         const std::map<std::string, std::string>& metadata) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = conversations_.find(conversation_id);
    if (it == conversations_.end()) {
        throw std::runtime_error("Conversation not found: " + conversation_id);
    }
    
    for (const auto& pair : metadata) {
        it->second->metadata[pair.first] = pair.second;
    }
}

void ConversationManager::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    conversations_.clear();
}

size_t ConversationManager::count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return conversations_.size();
}

} // namespace lm
