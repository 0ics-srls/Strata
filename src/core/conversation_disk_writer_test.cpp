// CPU-only tests for the conversation disk's background writer (ConversationDiskWriter, conversation_disk.hpp): the
// writes go to a thread of their own, a queued conversation is taken back from RAM, flush() waits for everything.
// No waits on the clock: the writer is held at a gate (its before_write hook) and the test waits on its waiting() count.
// Built with -DSTRATA_BUILD_CONVERSATION_TESTS=ON; no GPU, no model.
#include "strata/core/conversation_disk.hpp"
#include "conversation_disk_test_images.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <future>
#include <string>
#include <thread>
#include <vector>

using namespace strata::core;
using namespace disk_test;

namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

// Holds the writer before each write until open() - the "slow disk" - and says when a write started.
struct Gate {
    std::promise<void> opened;
    std::shared_future<void> open_f = opened.get_future().share();
    std::atomic<int> started{0};
    void hold(const SavedConversation&) { ++started; open_f.wait(); }
    void open() { opened.set_value(); }
};

void spin_until(const std::function<bool()>& done) {   // an event, not a time: the other thread changes what done() reads
    while (!done()) std::this_thread::yield();
}

const SessionFileIdentity kId{0x1111, 0x2222};
}  // namespace

int main() {
    std::string err;
    {   // submit() returns at once while the write is held; flush() waits for it; done() reports it
        const fs::path d = fresh_dir("writer-submit");
        ConversationDisk disk;
        check(disk.open(d.string(), 1ull << 32, kId, err), "open");
        Gate gate;
        std::atomic<int> done_calls{0};
        std::atomic<bool> done_stored{false};
        std::atomic<size_t> done_tokens{0};
        ConversationDiskWriter w(disk, 1,
            [&](bool stored, size_t tokens, double, const std::string&) { done_stored = stored; done_tokens = tokens; ++done_calls; },
            [&](const SavedConversation& s) { gate.hold(s); });
        w.submit(conversation(1000));
        spin_until([&] { return gate.started.load() == 1; });
        check(disk.size() == 0 && w.pending() == 0, "submit() came back while its write is held: nothing on disk yet");
        gate.open();
        w.flush();
        check(disk.size() == 1 && disk.best(prompt_of(1000, 1000, 9000), {}, true).tokens == 1000,
              "after flush() the conversation is on disk");
        check(done_calls == 1 && done_stored && done_tokens == 1000, "done() reported it stored, with its 1000 tokens");
        fs::remove_all(d);
    }
    {   // a queued conversation is taken back from RAM, whole, and is then not written; at_least is honoured
        const fs::path d = fresh_dir("writer-takeback");
        ConversationDisk disk;
        check(disk.open(d.string(), 1ull << 32, kId, err), "open");
        Gate gate;
        std::atomic<int> done_calls{0};
        ConversationDiskWriter w(disk, 1, [&](bool, size_t, double, const std::string&) { ++done_calls; },
                                 [&](const SavedConversation& s) { gate.hold(s); });
        w.submit(conversation(1000));                       // held at the gate: the one being written
        spin_until([&] { return gate.started.load() == 1; });
        w.submit(conversation(2000));                       // queued
        check(w.pending() == 1, "one being written, one queued");
        SavedConversation out;
        ConversationDisk::Match m;
        check(!w.take_pending(prompt_of(2000, 1000, 9000), {}, true, 1001, out, m) && w.pending() == 1,
              "take_pending() leaves a conversation that covers fewer than at_least tokens");
        check(w.take_pending(prompt_of(2000, 1000, 9000), {}, true, 1000, out, m) && m.tokens == 1000 && m.live,
              "a queued conversation is taken back with its match");
        check(!out.live.ids.empty() && out.live.ids[0] == 2000 && out.kv.size() == 2, "whole: its live state and its K/V");
        check(w.pending() == 0, "and it left the queue");
        gate.open();
        w.flush();
        check(disk.size() == 1 && disk.best(prompt_of(2000, 1000, 9000), {}, true).tokens == 0 && done_calls == 1,
              "a conversation taken back is not written");
        fs::remove_all(d);
    }
    {   // the conversation being written: take_pending() waits for the write and says false; then it is on disk
        const fs::path d = fresh_dir("writer-inflight");
        ConversationDisk disk;
        check(disk.open(d.string(), 1ull << 32, kId, err), "open");
        Gate gate;
        ConversationDiskWriter w(disk, 1, {}, [&](const SavedConversation& s) { gate.hold(s); });
        w.submit(conversation(1000));
        spin_until([&] { return gate.started.load() == 1; });
        std::atomic<int> result{-1};
        std::thread t([&] {
            SavedConversation out;
            ConversationDisk::Match m;
            result = w.take_pending(prompt_of(1000, 1000, 9000), {}, true, 1000, out, m) ? 1 : 0;
        });
        spin_until([&] { return w.waiting() == 1; });       // it is waiting on the write
        check(result == -1, "while the write is held, take_pending() does not return");
        gate.open();
        t.join();
        check(result == 0, "take_pending() of the conversation being written returns false");
        check(disk.best(prompt_of(1000, 1000, 9000), {}, true).tokens == 1000, "and when it returns the conversation is on disk");
        fs::remove_all(d);
    }
    {   // the queue holds at most max_pending: one more submit waits for a place
        const fs::path d = fresh_dir("writer-bound");
        ConversationDisk disk;
        check(disk.open(d.string(), 1ull << 32, kId, err), "open");
        Gate gate;
        ConversationDiskWriter w(disk, 1, {}, [&](const SavedConversation& s) { gate.hold(s); });
        w.submit(conversation(1000));
        spin_until([&] { return gate.started.load() == 1; });
        w.submit(conversation(2000));
        std::atomic<bool> third_in{false};
        std::thread t([&] { w.submit(conversation(3000)); third_in = true; });
        spin_until([&] { return w.waiting() == 1; });
        check(!third_in && w.pending() == 1, "a third submit waits: the queue never holds more than max_pending");
        gate.open();
        t.join();
        w.flush();
        check(third_in && disk.size() == 3, "once a place frees up it goes in, and all three are written");
        fs::remove_all(d);
    }
    {   // the destructor writes everything still queued
        const fs::path d = fresh_dir("writer-dtor");
        ConversationDisk disk;
        check(disk.open(d.string(), 1ull << 32, kId, err), "open");
        {
            ConversationDiskWriter w(disk, 1);
            w.submit(conversation(1000));
            w.submit(conversation(2000));
        }
        check(disk.size() == 2, "the destructor flushed both");
        fs::remove_all(d);
    }
    std::printf("conversation_disk_writer_test: %d checks passed\n", checks);
    return 0;
}
