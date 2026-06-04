#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

inline uint32_t fnv1a_hash(const std::string& key) {
    uint32_t hash = 2166136261u;
    for (unsigned char byte : key) {
        hash ^= byte;
        hash *= 16777619u;
    }
    return hash;
}

inline std::size_t assign_shard(const std::string& id, std::size_t num_shards) {
    return static_cast<std::size_t>(fnv1a_hash(id)) % num_shards;
}
