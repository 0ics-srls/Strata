// 0ics: the conversation cache's second tier, on disk.  A parked conversation that the RAM budget evicts is written as a
// session file (conversation_file.hpp) in its own directory instead of being dropped; a request whose prompt starts
// with one of them loads it back instead of reading the prompt again.  Reading 3 GB from a Mac's SSD takes well under
// a second, reading the same 200k-token prompt again about a minute.
//
// Each entry is two files: conv-<n>.bin (the session file: identity-checked and hashed on read) and conv-<n>.keys (its
// token keys, written after the session file: an entry without keys was never finished and is not indexed).  The keys
// carry the identity the file was written with, the steering mode, and the token ids and pictures of the live state and
// of the one checkpoint kept (the deepest, as a SAVE keeps: earlier ones only serve edits further back and cost a
// running state each).  The directory is the tier's own: it removes only conv-*.bin / conv-*.keys files with its
// identity - the least recently stored first, once `cap_bytes` is passed, and an entry it hands back with take() (that
// conversation is live again; a later eviction writes it again).  CPU only: no GPU, no model.
#pragma once

#include "strata/core/conversation_cache.hpp"
#include "strata/core/conversation_file.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

class ConversationDisk {
public:
    struct Match {
        size_t index = SIZE_MAX;   // the entry; SIZE_MAX: none
        int64_t tokens = 0;        // the prefix it covers (0: none)
        bool live = false;         // the entry's live state (else its checkpoint): where a restore goes on from
    };

    /// Indexes the entries `dir` holds for `id` (created when missing).  `cap_bytes`: the most both files of every
    /// entry may take together.  False (with `error`) when the directory cannot be made or listed.
    bool open(const std::string& dir, uint64_t cap_bytes, const SessionFileIdentity& id, std::string& error);
    bool enabled() const { return !dir_.empty(); }
    size_t size() const { return entries_.size(); }
    uint64_t bytes() const;

    /// Writes `image` (its live state, the deepest checkpoint and every K/V layer) as a new entry, then removes the
    /// oldest entries past the cap - the new one too, when it alone is larger.  False, with nothing written, for an
    /// image with pictures (their embeddings are not in the file) or a write failure.
    bool store(SavedConversation&& image, std::string& error, const SessionWriteOptions& options = {});

    /// The entry whose live state or checkpoint is the longest prefix of `prompt` (same pictures below it, same
    /// steering): the rule ConversationCache::best uses (conversation_prefix).
    template<class Token>
    Match best(const std::vector<Token>& prompt, const std::vector<ConversationImageKey>& images, bool cvec) const {
        Match m;
        for (size_t i = 0; i < entries_.size(); ++i) {
            if (entries_[i].cvec != cvec) continue;
            for (size_t j = 0; j < entries_[i].keys.size(); ++j) {
                const int64_t n = conversation_prefix(entries_[i].keys[j], prompt, images);
                if (n > m.tokens) m = {i, n, j == 0};
            }
        }
        return m;
    }

    /// Reads entry `index` into `image` (session_file_read: identity, limits, payload hash) and removes its files.
    /// On failure `image` is unchanged and the entry is dropped from the index (its files too when they are damaged).
    bool take(size_t index, SavedConversation& image, const SessionReadLimits& limits, std::string& error);

private:
    struct Entry {
        uint64_t seq = 0;                           // store order: the cap removes the lowest first
        uint64_t file_bytes = 0, keys_bytes = 0;
        bool cvec = true;
        std::vector<ConversationCheckpoint> keys;   // ids and pictures only: the live state, then the checkpoint
    };
    std::string path_of(uint64_t seq, const char* ext) const;
    void remove_entry(size_t index);
    void enforce_cap();

    std::string dir_;
    uint64_t cap_ = 0, next_seq_ = 1;
    SessionFileIdentity id_;
    std::vector<Entry> entries_;   // in store order
};

}  // namespace strata::core
