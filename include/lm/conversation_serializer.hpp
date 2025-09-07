// include/lm/conversation_serialization.hpp
#pragma once

#include "conversation.hpp"
#include <cereal/types/vector.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/chrono.hpp>

namespace lm {

template <class Archive>
void serialize(Archive& archive, ConversationTurn& turn) {
    archive(
        cereal::make_nvp("speaker", static_cast<int&>(turn.speaker)),
        cereal::make_nvp("text", turn.text),
        cereal::make_nvp("tokens", turn.tokens),
        cereal::make_nvp("timestamp", turn.timestamp),
        cereal::make_nvp("metadata", turn.metadata)
    );
}

template <class Archive>
void serialize(Archive& archive, Conversation& conv) {
    archive(
        cereal::make_nvp("turns", conv.turns),
        cereal::make_nvp("domain", conv.domain),
        cereal::make_nvp("language", conv.language),
        cereal::make_nvp("metadata", conv.metadata),
        cereal::make_nvp("start_time", conv.start_time),
        cereal::make_nvp("end_time", conv.end_time)
    );
}

} // namespace lm

