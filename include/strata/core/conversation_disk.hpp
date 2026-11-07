// 0ics: the conversation cache's second tier, on disk.  A parked conversation that the RAM budget evicts is written as a
// session file (conversation_file.hpp) in its own directory instead of being dropped; a request whose prompt starts
// with one of them loads it back instead of reading the prompt again.  Reading 3 GB from a Mac's SSD takes well under
// a second, reading the same 200k-token prompt again about a minute.
//
// Each entry is two files: conv-<n>.bin (the session file: identity-checked and hashed on read) and conv-<n>.keys (its
// token keys, written after the session file: an entry without keys was never finished and is not indexed).  The keys
// carry the identity the file was written with, the steering mode, and the token ids and pictures of the live state and
// of the one checkpoint kept (the deepest, as a SAVE keeps: earlier ones only serve edits further back and cost a
// running state each).  The directory holds every model's entries: each model indexes, matches and reads back only its
// own (its LRU), but `cap_bytes` counts the complete entries of every model, and past it the entry with the lowest store
// number goes, whatever its model (0ics, ubuntu-ai: one cap for the models that share the folder).  The store number is
// common to the directory and is the recency: an entry handed back with take() leaves the disk (that conversation is live
// again) and a later eviction writes it again with a new number.  It removes only conv-*.bin / conv-*.keys files; another
// model's unfinished entry (no readable keys, or a session file of another size) is neither counted nor touched.  CPU
// only: no GPU, no model.
#pragma once

#include "strata/core/conversation_cache.hpp"
#include "strata/core/conversation_file.hpp"

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace strata::core {

class ConversationDisk {
public:
    struct Match {
        size_t index = SIZE_MAX;   // the entry's position when best() ran (informative: it shifts as entries go)
        uint64_t seq = 0;          // the entry's store number: what take() looks up (0: none)
        int64_t tokens = 0;        // the prefix it covers (0: none)
        bool live = false;         // the entry's live state (else its checkpoint): where a restore goes on from
    };

    /// The longest prefix of `prompt` among `keys` (the live state first, then the kept checkpoint), with the same
    /// steering and the same pictures below it - the rule ConversationCache::best uses (conversation_prefix).  The one
    /// comparison the disk and the writer's queue (ConversationDiskWriter) share.  `index` and `seq` stay unset.
    template<class Token>
    static Match match(const std::vector<ConversationCheckpoint>& keys, bool keys_cvec, const std::vector<Token>& prompt,
                       const std::vector<ConversationImageKey>& images, bool cvec) {
        Match m;
        if (keys_cvec != cvec) return m;
        for (size_t j = 0; j < keys.size(); ++j) {
            const int64_t n = conversation_prefix(keys[j], prompt, images);
            if (n > m.tokens) { m.tokens = n; m.live = j == 0; }
        }
        return m;
    }

    /// Indexes the entries `dir` holds for `id` and counts every model's complete entries (created when missing).
    /// `cap_bytes`: the most both files of every entry, of every model, may take together.  False (with `error`) when the
    /// directory cannot be made or listed.
    bool open(const std::string& dir, uint64_t cap_bytes, const SessionFileIdentity& id, std::string& error);
    bool enabled() const { return !dir_.empty(); }
    size_t size() const;           ///< this model's entries
    uint64_t bytes() const;        ///< this model's entries' files
    uint64_t total_bytes() const;  ///< every model's complete entries: what the cap counts

    /// Writes `image` (its live state, the deepest checkpoint and every K/V layer) as a new entry, then removes the
    /// oldest entries past the cap - the new one too, when it alone is larger.  False, with nothing written, for an
    /// image with pictures (their embeddings are not in the file) or a write failure.
    bool store(SavedConversation&& image, std::string& error, const SessionWriteOptions& options = {});

    /// The entry whose live state or checkpoint is the longest prefix of `prompt` (same pictures below it, same
    /// steering): the rule ConversationCache::best uses (conversation_prefix).
    template<class Token>
    Match best(const std::vector<Token>& prompt, const std::vector<ConversationImageKey>& images, bool cvec) const {
        const std::lock_guard<std::mutex> lock(mu_);
        Match m;
        for (size_t i = 0; i < entries_.size(); ++i) {
            if (!entries_[i].own) continue;
            const Match e = match(entries_[i].keys, entries_[i].cvec, prompt, images, cvec);
            if (e.tokens > m.tokens) m = {i, entries_[i].seq, e.tokens, e.live};
        }
        return m;
    }

    /// Reads the entry `m` names (by its store number: another thread may have removed older entries meanwhile) into
    /// `image` (session_file_read: identity, limits, payload hash) and removes its files.  False with "conversation disk:
    /// entry gone" when it is no longer there.  On a read refused for RAM the entry goes back in the index (its files
    /// stay); on any other failure it leaves the index and its files go (damaged).
    bool take(const Match& m, SavedConversation& image, const SessionReadLimits& limits, std::string& error);

private:
    struct Entry {
        uint64_t seq = 0;                           // store order, common to every model: the cap removes the lowest first
        uint64_t file_bytes = 0, keys_bytes = 0;
        bool own = true;                            // false: another model's entry - counted and evicted, never read
        bool cvec = true;
        std::vector<ConversationCheckpoint> keys;   // own entries only: the live state, then the checkpoint
    };
    std::string path_of(uint64_t seq, const char* ext) const;
    void remove_files(uint64_t seq) const;
    void remove_entry(size_t index);   // with mu_ held
    void enforce_cap();                // with mu_ held
    uint64_t total_bytes_locked() const;

    // Safe across threads (0ics: the writer thread stores while a request looks up and takes back): mu_ guards entries_
    // and next_seq_; the session files are written and read outside it.
    mutable std::mutex mu_;
    std::string dir_;
    uint64_t cap_ = 0, next_seq_ = 1;
    SessionFileIdentity id_;
    std::vector<Entry> entries_;   // every model's complete entries, in store order
};

/// 0ics: the disk tier's writes on a thread of their own, so a request does not wait for the conversation that leaves
/// (2.4 s for 36k tokens, ~10 s for 200k on ubuntu-ai's SATA SSD).  One thread; at most `max_pending` conversations wait
/// beside the one being written (a bound on RAM: one more submit waits for a place).  A request looks in the queue first:
/// a queued conversation that covers its prompt is taken back from RAM (take_pending) and never written; if it is the one
/// being written, take_pending waits for the write and says false (the request then finds it on disk).  flush() waits
/// until everything submitted is written (the engine stopping); the destructor flushes.  `done` runs on the writer's
/// thread after each write; `before_write` (tests) runs there just before it.
class ConversationDiskWriter {
public:
    using Done = std::function<void(bool stored, size_t tokens, double ms, const std::string& why)>;
    using BeforeWrite = std::function<void(const SavedConversation&)>;

    explicit ConversationDiskWriter(ConversationDisk& disk, size_t max_pending = 1, Done done = {},
                                    BeforeWrite before_write = {});
    ~ConversationDiskWriter();
    ConversationDiskWriter(const ConversationDiskWriter&) = delete;
    ConversationDiskWriter& operator=(const ConversationDiskWriter&) = delete;

    void submit(SavedConversation&& image);   ///< waits only while max_pending conversations are already queued

    /// A queued conversation whose live state or kept checkpoint covers at least `at_least` tokens of `prompt` (the
    /// longest): out of the queue into `out`, with its match.  The one being written: waits for it, then false.
    template<class Token>
    bool take_pending(const std::vector<Token>& prompt, const std::vector<ConversationImageKey>& images, bool cvec,
                      int64_t at_least, SavedConversation& out, ConversationDisk::Match& m) {
        std::unique_lock<std::mutex> lock(mu_);
        size_t best = SIZE_MAX;
        ConversationDisk::Match bm;
        for (size_t i = 0; i < queue_.size(); ++i) {
            const auto e = ConversationDisk::match(queue_[i].keys, queue_[i].image.cvec, prompt, images, cvec);
            if (e.tokens >= at_least && e.tokens > bm.tokens) { bm = e; best = i; }
        }
        if (best != SIZE_MAX) {
            out = std::move(queue_[best].image);
            queue_.erase(queue_.begin() + (std::ptrdiff_t) best);
            m = bm;
            cv_.notify_all();   // a place is free
            return true;
        }
        if (writing_ && ConversationDisk::match(writing_keys_, writing_cvec_, prompt, images, cvec).tokens >= at_least) {
            ++waiting_;
            cv_.wait(lock, [&] { return !writing_; });
            --waiting_;
        }
        return false;
    }

    void flush();             ///< until the queue is empty and nothing is being written
    size_t pending() const;   ///< queued, not counting the one being written
    size_t waiting() const;   ///< threads waiting in submit (for a place) or take_pending (for a write): tests

private:
    struct Queued {
        SavedConversation image;
        std::vector<ConversationCheckpoint> keys;   // ids and pictures only: the live state, then the kept checkpoint
    };
    static std::vector<ConversationCheckpoint> keys_of(const SavedConversation& image);
    void run();

    ConversationDisk& disk_;
    size_t max_pending_;
    Done done_;
    BeforeWrite before_write_;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::deque<Queued> queue_;
    std::vector<ConversationCheckpoint> writing_keys_;
    bool writing_cvec_ = true, writing_ = false, stop_ = false;
    size_t waiting_ = 0;
    std::thread thread_;
};

/// 0ics: the tier's default cap (--conversation-disk-gib): 100 GB for every model in the folder (ubuntu-ai, 7/11)
constexpr int64_t kConversationDiskDefaultGib = 100;
/// 0ics: a read back moves a whole conversation (0.3-4 GB) for the tokens it saves: worth it from 2,048 of them (a few
/// hundred ms of prompt reading); a prompt that only shares a chat template's first tokens is read instead
constexpr int64_t kConversationDiskMinGain = 2048;

/// 0ics: the engine options the tier depends on.  The peer (--peer-device) holds experts only - the conversation's K/V and
/// state stay on CUDA0 - so it is allowed; --batch is not (this branch does not park the batch slots).
struct ConversationDiskSetup {
    bool serve = false;
    int prompt_cache = 0;
    bool layer_split = false;   ///< --layer-split, or several GPUs as stages
    bool batch = false;         ///< --batch > 0
    int peer_device = 0;
};

/// Empty when the tier can open with these options, else why not (the engine prints it).
inline std::string conversation_disk_refusal(const ConversationDiskSetup& s) {
    if (!s.serve) return "it needs --serve";
    if (s.prompt_cache <= 0) return "it needs --prompt-cache > 0";
    if (s.layer_split) return "it does not support --layer-split";
    if (s.batch) return "it does not support --batch";
    return {};
}

/// A conversation on disk is read back only when it covers at least kConversationDiskMinGain more tokens than the best
/// source already at hand (the live session, a parked conversation, a batch slot).
inline bool conversation_disk_worth(int64_t disk_tokens, int64_t best_elsewhere) {
    return disk_tokens >= best_elsewhere + kConversationDiskMinGain;
}

/// 0ics: the client's output reserve plus a turn.  A conversation past max_context minus this is compacted by the client
/// before it could be resumed (OpenCode compacts at 262,144 - 32,768; after it the prompt is a new summary that no saved
/// conversation matches), so the disk does not keep it: ~21 KiB per token and seconds of writing for nothing.
constexpr int64_t kConversationDiskResumeReserve = 40960;

/// --conversation-disk-max-tokens N: N < 0 = max_context - kConversationDiskResumeReserve (at least 1: 0 means no
/// limit), N = 0 = no limit, N > 0 = N.
inline int64_t conversation_disk_max_tokens(int64_t option, int64_t max_context) {
    if (option >= 0) return option;
    return std::max<int64_t>(1, max_context - kConversationDiskResumeReserve);
}

/// A conversation of `tokens` is worth writing to disk under `max_tokens` (0 = no limit).
inline bool conversation_disk_keeps(int64_t tokens, int64_t max_tokens) {
    return max_tokens == 0 || tokens <= max_tokens;
}

}  // namespace strata::core
