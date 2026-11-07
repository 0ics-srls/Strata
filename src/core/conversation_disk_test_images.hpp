// 0ics: the synthetic conversations the disk tier's tests share (conversation_disk_test, conversation_disk_writer_test).
// CPU only: no GPU, no model.
#pragma once

#include "strata/core/conversation_disk.hpp"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

namespace disk_test {
using namespace strata::core;
namespace fs = std::filesystem;

inline ConversationBuffer pattern(size_t n, uint8_t seed) {
    ConversationBuffer b;
    b.resize(n);
    b.visit(0, n, [&](uint8_t* p, size_t c, size_t at) {
        for (size_t i = 0; i < c; ++i) p[i] = uint8_t((at + i) * 131u + seed);
        return true;
    });
    return b;
}

inline std::vector<uint8_t> bytes_of(size_t n, uint8_t seed) {
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = uint8_t(i * 7u + seed);
    return v;
}

// ids first, first+1, ...: two conversations with different `first` share no prefix
inline ConversationCheckpoint checkpoint(size_t tokens, int32_t first, uint8_t seed) {
    ConversationCheckpoint c;
    for (size_t i = 0; i < tokens; ++i) c.ids.push_back(first + int32_t(i));
    c.gdn = bytes_of(4099, seed);
    c.ple = bytes_of(77, seed + 1);
    c.tails = bytes_of(301, seed + 2);
    c.dead = bytes_of(64, seed + 3);
    c.used = 42 + seed;
    return c;
}

inline SavedConversation conversation(int32_t first, size_t kv_bytes = 4096) {
    SavedConversation s;
    for (size_t i = 0; i < s.geometry.size(); ++i) s.geometry[i] = int64_t(100 + i);
    s.layer_lo = 0; s.layer_hi = 48;
    s.cvec = true;
    s.live = checkpoint(1000, first, 1);
    s.checkpoints.push_back(checkpoint(200, first, 2));
    s.checkpoints.push_back(checkpoint(500, first, 3));   // the deepest: the only one a disk entry keeps
    for (int layer = 0; layer < 2; ++layer) {
        ConversationKv kv;
        kv.format = 1; kv.cells = 1000; kv.heads = 2; kv.head_dim = 256; kv.page_size = 4; kv.pooled_rows = 251;
        kv.idx_dim = 128;
        kv.k = pattern(kv_bytes, uint8_t(layer));
        kv.v = pattern(kv_bytes, uint8_t(layer + 10));
        kv.k_scale = pattern(512, uint8_t(layer + 20));
        kv.v_scale = pattern(512, uint8_t(layer + 30));
        kv.pooled = pattern(251 * 128 * 4, uint8_t(layer + 40));
        s.kv.push_back(std::move(kv));
    }
    return s;
}

inline std::vector<int32_t> prompt_of(int32_t first, size_t same, int32_t then) {
    std::vector<int32_t> p;
    for (size_t i = 0; i < same; ++i) p.push_back(first + int32_t(i));
    for (int i = 0; i < 40; ++i) p.push_back(then + i);
    return p;
}

inline size_t files_in(const fs::path& d) {
    size_t n = 0;
    for (const auto& e : fs::directory_iterator(d)) n += e.is_regular_file() ? 1 : 0;
    return n;
}

inline fs::path fresh_dir(const char* tag) {
    const fs::path d = fs::temp_directory_path() / (std::string("strata-disk-test-") + tag + "-" + std::to_string((long long) std::rand()));
    fs::create_directories(d);
    return d;
}
}  // namespace disk_test
