// CPU-only tests for the conversation cache's disk tier (include/strata/core/conversation_disk.hpp).
// Built with -DSTRATA_BUILD_CONVERSATION_TESTS=ON; no GPU, no model.
#include "strata/core/conversation_disk.hpp"
#include "conversation_disk_test_images.hpp"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

using namespace strata::core;
using namespace disk_test;
namespace fs = std::filesystem;

namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
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
        check(d.take(m, got, {}, err), "take the entry");
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
        check(probe.take(probe.best(prompt_of(1000, 1000, 9000), {}, true), taken, {}, err), "empty the probe");
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
    // 0ics: safe across threads (a writer thread stores while the request thread looks up and takes back)
    {   // take() finds its entry by store number, even after the cap removed an older one in between
        const fs::path d = fresh_dir("seq");
        ConversationDisk disk;
        check(disk.open(d.string(), 2 * one + one / 2, ida, err) && disk.store(conversation(1000), err) &&
              disk.store(conversation(2000), err), "two stored, room for two and a half");
        const auto m_old = disk.best(prompt_of(1000, 1000, 9000), {}, true);
        const auto m_b = disk.best(prompt_of(2000, 1000, 9000), {}, true);
        check(m_b.tokens == 1000 && m_b.seq != 0 && m_b.seq != m_old.seq, "a match carries its entry's store number");
        check(disk.store(conversation(3000), err) && disk.size() == 2, "a third: the cap removes the oldest, the positions shift");
        SavedConversation got;
        check(disk.take(m_b, got, {}, err) && !got.live.ids.empty() && got.live.ids[0] == 2000,
              "take() gets the matched conversation by its store number, not by its old position");
        SavedConversation none;
        err.clear();
        check(!disk.take(m_old, none, {}, err) && err.find("entry gone") != std::string::npos,
              "an entry the cap removed meanwhile: take() is false and says the entry is gone");
        check(disk.size() == 1 && disk.best(prompt_of(3000, 1000, 9000), {}, true).tokens == 1000, "the others are untouched");
        fs::remove_all(d);
    }
    {   // match() on an entry's keys gives what best() gives
        const fs::path d = fresh_dir("match");
        ConversationDisk disk;
        check(disk.open(d.string(), 1ull << 30, ida, err) && disk.store(conversation(1000), err), "one stored");
        const SavedConversation s = conversation(1000);
        const std::vector<ConversationCheckpoint> keys{s.live, s.checkpoints.back()};   // the live state, the deepest checkpoint
        struct Case { std::vector<int32_t> prompt; std::vector<ConversationImageKey> imgs; bool cvec; };
        const std::vector<Case> cases{{prompt_of(1000, 1000, 9000), {}, true}, {prompt_of(1000, 700, 9000), {}, true},
                                      {prompt_of(1000, 300, 9000), {}, true}, {prompt_of(5000, 1000, 9000), {}, true},
                                      {prompt_of(1000, 1000, 9000), {}, false}, {prompt_of(1000, 1000, 9000), {{10, 0x99}}, true}};
        bool same = true;
        for (const Case& c : cases) {
            const auto a = disk.best(c.prompt, c.imgs, c.cvec);
            const auto b = ConversationDisk::match(keys, true, c.prompt, c.imgs, c.cvec);
            same = same && a.tokens == b.tokens && (a.tokens == 0 || a.live == b.live);
        }
        check(same, "match() on the keys equals best(): prefix, checkpoint, steering, pictures");
        fs::remove_all(d);
    }
    {   // two threads: one stores 20 conversations while the other looks them up and takes them back
        const fs::path d = fresh_dir("threads");
        ConversationDisk disk;
        check(disk.open(d.string(), 1ull << 32, ida, err), "open for two threads");
        constexpr int n = 20;
        std::atomic<bool> stored_all{false}, store_failed{false};
        std::thread writer([&] {
            std::string e;
            for (int i = 0; i < n; ++i)
                if (!disk.store(conversation(10000 + i * 1000, 512), e)) store_failed = true;
            stored_all = true;
        });
        std::vector<int> taken(n, 0);
        int left = n;
        bool take_failed = false;
        while (left > 0) {
            const bool last_round = stored_all.load();
            for (int i = 0; i < n; ++i) {
                if (taken[i]) continue;
                const auto m = disk.best(prompt_of(10000 + i * 1000, 1000, 900000), {}, true);
                if (m.tokens != 1000) continue;
                SavedConversation got;
                std::string e;
                if (disk.take(m, got, {}, e) && !got.live.ids.empty() && got.live.ids[0] == 10000 + i * 1000) { ++taken[i]; --left; }
                else take_failed = true;
            }
            if (last_round && left > 0) break;   // everything was stored before this round and some was not found
        }
        writer.join();
        check(!store_failed && !take_failed, "no store and no take failed with two threads");
        check(left == 0 && std::all_of(taken.begin(), taken.end(), [](int t) { return t == 1; }),
              "every conversation was found and taken back exactly once");
        check(disk.size() == 0 && files_in(d) == 0, "at the end the index and the files agree: empty");
        fs::remove_all(d);
    }
    fs::remove_all(dir);
    std::printf("conversation_disk_test: %d checks passed\n", checks);
    return 0;
}
