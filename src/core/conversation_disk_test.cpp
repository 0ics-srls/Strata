// CPU-only tests for the conversation cache's disk tier (include/strata/core/conversation_disk.hpp).
// Built with -DSTRATA_BUILD_CONVERSATION_TESTS=ON; no GPU, no model.
#include "strata/core/conversation_disk.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

using namespace strata::core;
namespace fs = std::filesystem;

namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

ConversationBuffer pattern(size_t n, uint8_t seed) {
    ConversationBuffer b;
    b.resize(n);
    b.visit(0, n, [&](uint8_t* p, size_t c, size_t at) {
        for (size_t i = 0; i < c; ++i) p[i] = uint8_t((at + i) * 131u + seed);
        return true;
    });
    return b;
}

std::vector<uint8_t> bytes_of(size_t n, uint8_t seed) {
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = uint8_t(i * 7u + seed);
    return v;
}

// ids first, first+1, ...: two conversations with different `first` share no prefix
ConversationCheckpoint checkpoint(size_t tokens, int32_t first, uint8_t seed) {
    ConversationCheckpoint c;
    for (size_t i = 0; i < tokens; ++i) c.ids.push_back(first + int32_t(i));
    c.gdn = bytes_of(4099, seed);
    c.ple = bytes_of(77, seed + 1);
    c.tails = bytes_of(301, seed + 2);
    c.dead = bytes_of(64, seed + 3);
    c.used = 42 + seed;
    return c;
}

SavedConversation conversation(int32_t first, size_t kv_bytes = 4096) {
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

std::vector<int32_t> prompt_of(int32_t first, size_t same, int32_t then) {
    std::vector<int32_t> p;
    for (size_t i = 0; i < same; ++i) p.push_back(first + int32_t(i));
    for (int i = 0; i < 40; ++i) p.push_back(then + i);
    return p;
}

size_t files_in(const fs::path& d) {
    size_t n = 0;
    for (const auto& e : fs::directory_iterator(d)) n += e.is_regular_file() ? 1 : 0;
    return n;
}

bool same_kv(const ConversationKv& x, const ConversationKv& y) {
    return x.format == y.format && x.cells == y.cells && x.heads == y.heads && x.head_dim == y.head_dim &&
           x.page_size == y.page_size && x.pooled_rows == y.pooled_rows && x.idx_dim == y.idx_dim && x.k == y.k &&
           x.v == y.v && x.k_scale == y.k_scale && x.v_scale == y.v_scale && x.pooled == y.pooled;
}
}  // namespace

int main() {
    const fs::path dir = fs::temp_directory_path() / ("strata-disk-test-" + std::to_string((long long) std::rand()));
    fs::create_directories(dir);
    const SessionFileIdentity id{0x1111, 0x2222};
    std::string err;

    {   // empty directory: nothing to find
        ConversationDisk d;
        check(d.open(dir.string(), 1ull << 30, id, err), "open an empty directory");
        check(d.enabled() && d.size() == 0 && d.bytes() == 0, "empty tier");
        check(d.best(prompt_of(1000, 1000, 9000), {}, true).tokens == 0, "no match in an empty tier");
    }
    {   // store, then find by the live state and by the deepest checkpoint
        ConversationDisk d;
        check(d.open(dir.string(), 1ull << 30, id, err), "open");
        check(d.store(conversation(1000), err), "store a conversation");
        check(d.size() == 1 && d.bytes() > 0, "one entry stored");
        check(files_in(dir) == 2, "a session file and its keys");
        check(d.best(prompt_of(1000, 1000, 9000), {}, true).tokens == 1000, "the live state is a prefix");
        check(d.best(prompt_of(1000, 1000, 9000), {}, true).live, "and the match says it is the live state");
        check(d.best(prompt_of(1000, 700, 9000), {}, true).tokens == 500, "diverged after 700: the deepest checkpoint");
        check(!d.best(prompt_of(1000, 700, 9000), {}, true).live, "and the match says it is a checkpoint");
        check(d.best(prompt_of(1000, 300, 9000), {}, true).tokens == 0, "the shallower checkpoint is not kept");
        check(d.best(prompt_of(5000, 1000, 9000), {}, true).tokens == 0, "another conversation");
        check(d.best(prompt_of(1000, 1000, 9000), {}, false).tokens == 0, "another steering (cvec)");
        check(d.best(prompt_of(1000, 1000, 9000), {{10, 0x99}}, true).tokens == 0, "a picture the entry does not have");
    }
    {   // the index comes back from the keys after a restart; take() gives the image back and frees the disk
        ConversationDisk d;
        check(d.open(dir.string(), 1ull << 30, id, err), "reopen");
        check(d.size() == 1, "the stored entry is found again");
        const auto m = d.best(prompt_of(1000, 1000, 9000), {}, true);
        check(m.tokens == 1000, "found again by its live state");
        SavedConversation got;
        check(d.take(m.index, got, {}, err), "take the entry");
        const SavedConversation want = conversation(1000);
        check(got.live.ids == want.live.ids && got.live.gdn == want.live.gdn && got.cvec == want.cvec,
              "the live state comes back");
        check(got.checkpoints.size() == 1 && got.checkpoints[0].ids == want.checkpoints[1].ids,
              "only the deepest checkpoint comes back");
        check(got.kv.size() == 2 && same_kv(got.kv[0], want.kv[0]) && same_kv(got.kv[1], want.kv[1]),
              "the K/V comes back byte for byte");
        check(d.size() == 0 && files_in(dir) == 0, "a taken entry leaves the disk (it is live again)");
    }
    {   // the cap: past it the oldest entry goes; an image with pictures is not stored
        ConversationDisk probe;
        check(probe.open(dir.string(), 1ull << 30, id, err) && probe.store(conversation(1000), err), "probe size");
        const uint64_t one = probe.bytes();
        SavedConversation taken;
        check(probe.take(0, taken, {}, err), "empty the probe");
        ConversationDisk d;
        check(d.open(dir.string(), one + one / 2, id, err), "open with room for one entry and a half");
        check(d.store(conversation(1000), err) && d.store(conversation(5000), err), "store two");
        check(d.size() == 1 && files_in(dir) == 2, "the cap keeps one");
        check(d.best(prompt_of(1000, 1000, 9000), {}, true).tokens == 0, "the oldest went");
        check(d.best(prompt_of(5000, 1000, 9000), {}, true).tokens == 1000, "the newest stays");
        SavedConversation pic = conversation(7000);
        pic.live.imgs = {{3, 0x1234}};
        check(!d.store(std::move(pic), err), "an image with pictures is not stored");
        check(d.size() == 1, "nothing added for it");
    }
    {   // keys of another model identity are left alone: not indexed, not removed
        ConversationDisk other;
        check(other.open(dir.string(), 1ull << 30, SessionFileIdentity{0x3333, 0x2222}, err), "open as another model");
        check(other.size() == 0, "another model's entries are not indexed");
        check(files_in(dir) == 2, "and not removed");
    }
    fs::remove_all(dir);
    std::printf("conversation_disk_test: %d checks passed\n", checks);
    return 0;
}
