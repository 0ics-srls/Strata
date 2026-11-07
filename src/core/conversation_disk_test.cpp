// CPU-only tests for the conversation cache's disk tier (include/strata/core/conversation_disk.hpp).
// Built with -DSTRATA_BUILD_CONVERSATION_TESTS=ON; no GPU, no model.
#include "strata/core/conversation_disk.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
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

// the highest n of the conv-<n>.keys files in d (0: none)
uint64_t max_seq(const fs::path& d) {
    uint64_t m = 0;
    for (const auto& e : fs::directory_iterator(d)) {
        const std::string n = e.path().filename().string();
        if (n.size() > 10 && n.compare(0, 5, "conv-") == 0 && n.compare(n.size() - 5, 5, ".keys") == 0)
            m = std::max<uint64_t>(m, std::stoull(n.substr(5, n.size() - 10)));
    }
    return m;
}

fs::path fresh_dir(const char* tag) {
    const fs::path d = fs::temp_directory_path() / (std::string("strata-disk-test-") + tag + "-" + std::to_string((long long) std::rand()));
    fs::create_directories(d);
    return d;
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
    // 0ics: one cap for every model in the directory.  Each model indexes, matches and reads back only its own entries
    // (its LRU); the cap counts every model's files and past it the least recent entry of all goes, whatever its model.
    // (One engine runs at a time: each step opens a fresh instance, as an engine start does.)
    const SessionFileIdentity ida{0x1111, 0x2222}, idb{0x3333, 0x2222};
    uint64_t one = 0;
    {
        const fs::path d = fresh_dir("one");
        ConversationDisk probe;
        check(probe.open(d.string(), 1ull << 30, ida, err) && probe.store(conversation(1000), err), "probe size (models)");
        one = probe.bytes();
        fs::remove_all(d);
    }
    const uint64_t cap3 = 3 * one + one / 2;   // room for three entries and a half
    {   // two models share the cap; the oldest of all goes
        const fs::path d = fresh_dir("share");
        {
            ConversationDisk a;
            check(a.open(d.string(), cap3, ida, err) && a.store(conversation(1000), err) && a.store(conversation(2000), err),
                  "A stores two");
        }
        {
            ConversationDisk b;
            check(b.open(d.string(), cap3, idb, err), "B opens the same directory");
            check(b.size() == 0 && b.bytes() == 0, "B has none of its own");
            check(b.total_bytes() == 2 * one, "but the total counts A's two entries");
            check(b.store(conversation(3000), err) && files_in(d) == 6, "B stores one: three entries, under the cap, none removed");
            check(b.store(conversation(4000), err), "B stores another, past the cap");
            check(files_in(d) == 6 && b.total_bytes() <= cap3, "one entry went and the total is under the cap");
            check(b.size() == 2, "B's own two stay");
        }
        {
            ConversationDisk a;
            check(a.open(d.string(), cap3, ida, err) && a.size() == 1, "A reopened finds its one remaining entry");
            check(a.best(prompt_of(1000, 1000, 9000), {}, true).tokens == 0, "A's oldest went: the least recent of all");
            check(a.best(prompt_of(2000, 1000, 9000), {}, true).tokens == 1000, "A's newer one stays");
            check(a.best(prompt_of(3000, 1000, 9000), {}, true).tokens == 0 && a.best(prompt_of(4000, 1000, 9000), {}, true).tokens == 0,
                  "A never matches B's entries");
        }
        fs::remove_all(d);
    }
    {   // a1, b1, a2 stored in this order; b2 passes the cap: a1 goes, not b1
        const fs::path d = fresh_dir("order");
        { ConversationDisk a; check(a.open(d.string(), cap3, ida, err) && a.store(conversation(1000), err), "a1"); }
        { ConversationDisk b; check(b.open(d.string(), cap3, idb, err) && b.store(conversation(3000), err), "b1"); }
        { ConversationDisk a; check(a.open(d.string(), cap3, ida, err) && a.store(conversation(2000), err), "a2"); }
        { ConversationDisk b; check(b.open(d.string(), cap3, idb, err) && b.store(conversation(4000), err), "b2 past the cap"); }
        {
            ConversationDisk a;
            check(a.open(d.string(), cap3, ida, err) && a.best(prompt_of(1000, 1000, 9000), {}, true).tokens == 0, "a1 went");
            check(a.best(prompt_of(2000, 1000, 9000), {}, true).tokens == 1000, "a2 stays");
            ConversationDisk b;
            check(b.open(d.string(), cap3, idb, err) && b.best(prompt_of(3000, 1000, 9000), {}, true).tokens == 1000,
                  "b1 stays: it is newer than a1");
            check(b.size() == 2, "b1 and b2");
        }
        {   // a reopen with a smaller cap removes the least recent of all until the total fits: b1 and a2 go, b2 stays
            const uint64_t before = max_seq(d);
            ConversationDisk a;
            check(a.open(d.string(), one + one / 2, ida, err), "reopen with room for one entry and a half");
            check(a.total_bytes() <= one + one / 2 && files_in(d) == 2, "one entry of all is left");
            check(a.size() == 0, "and it is not A's");
            ConversationDisk b;
            check(b.open(d.string(), 1ull << 30, idb, err) && b.best(prompt_of(4000, 1000, 9000), {}, true).tokens == 1000,
                  "it is b2, the most recent");
            ConversationDisk c;
            check(c.open(d.string(), 1ull << 30, ida, err) && c.store(conversation(5000), err), "A stores a new entry");
            check(max_seq(d) > before, "its store number is higher than every file in the directory, of any model");
        }
        fs::remove_all(d);
    }
    {   // another model's unfinished entry (a session file without its keys) is not counted and stays
        const fs::path d = fresh_dir("unfinished");
        {
            std::ofstream raw(d / "conv-500.bin", std::ios::binary);
            const std::string junk(1 << 16, 'x');
            raw.write(junk.data(), (std::streamsize) junk.size());
        }
        ConversationDisk a;
        check(a.open(d.string(), cap3, ida, err) && a.total_bytes() == 0, "an unfinished entry is not counted");
        check(a.store(conversation(1000), err) && a.store(conversation(2000), err) && a.store(conversation(3000), err) &&
              a.store(conversation(4000), err), "A stores four, past the cap");
        check(fs::exists(d / "conv-500.bin"), "the unfinished file stays");
        check(a.size() == 3 && a.total_bytes() <= cap3, "the cap holds over A's complete entries");
        fs::remove_all(d);
    }
    fs::remove_all(dir);
    std::printf("conversation_disk_test: %d checks passed\n", checks);
    return 0;
}
