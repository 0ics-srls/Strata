// 0ics: the conversation cache's disk tier (include/strata/core/conversation_disk.hpp).
#include "strata/core/conversation_disk.hpp"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <system_error>

namespace strata::core {
namespace fs = std::filesystem;

namespace {
constexpr char kKeysMagic[4] = {'S', '0', 'C', 'K'};
constexpr uint32_t kKeysVersion = 1;
constexpr uint64_t kMaxKeyTokens = 1ull << 24, kMaxKeyImages = 1ull << 16, kMaxKeys = 64;

template<class T> void put(std::string& out, T v) { out.append(reinterpret_cast<const char*>(&v), sizeof v); }
template<class T> bool get(std::ifstream& in, T& v) { return (bool) in.read(reinterpret_cast<char*>(&v), sizeof v); }

// The keys file: magic, version, the identity, the steering mode, the session file's size, then for each key (the
// live state, then the kept checkpoint) its token ids and its pictures.
std::string keys_bytes(const SessionFileIdentity& id, bool cvec, uint64_t file_bytes,
                       const std::vector<const ConversationCheckpoint*>& keys) {
    std::string out(kKeysMagic, sizeof kKeysMagic);
    put(out, kKeysVersion);
    put(out, id.model);
    put(out, id.config);
    put(out, (uint8_t) (cvec ? 1 : 0));
    put(out, file_bytes);
    put(out, (uint32_t) keys.size());
    for (const ConversationCheckpoint* k : keys) {
        put(out, (uint64_t) k->ids.size());
        out.append(reinterpret_cast<const char*>(k->ids.data()), k->ids.size() * sizeof(int32_t));
        put(out, (uint64_t) k->imgs.size());
        for (const ConversationImageKey& m : k->imgs) { put(out, m.start); put(out, m.hash); }
    }
    return out;
}

// false for anything that is not a complete keys file of version 1 (the entry is then not indexed)
bool read_keys(const fs::path& p, SessionFileIdentity& id, bool& cvec, uint64_t& file_bytes,
               std::vector<ConversationCheckpoint>& keys) {
    std::ifstream in(p, std::ios::binary);
    char magic[4];
    uint32_t version = 0, n = 0;
    uint8_t cv = 0;
    if (!in.read(magic, sizeof magic) || std::memcmp(magic, kKeysMagic, sizeof magic) != 0 || !get(in, version) ||
        version != kKeysVersion || !get(in, id.model) || !get(in, id.config) || !get(in, cv) || !get(in, file_bytes) ||
        !get(in, n) || n == 0 || n > kMaxKeys)
        return false;
    cvec = cv != 0;
    keys.assign(n, {});
    for (ConversationCheckpoint& k : keys) {
        uint64_t ni = 0, nm = 0;
        if (!get(in, ni) || ni > kMaxKeyTokens) return false;
        k.ids.resize(ni);
        if (ni && !in.read(reinterpret_cast<char*>(k.ids.data()), (std::streamsize) (ni * sizeof(int32_t)))) return false;
        if (!get(in, nm) || nm > kMaxKeyImages) return false;
        k.imgs.resize(nm);
        for (ConversationImageKey& m : k.imgs)
            if (!get(in, m.start) || !get(in, m.hash)) return false;
    }
    return in.peek() == std::char_traits<char>::eof();
}

// conv-<n>.keys -> n (0: not one of this tier's names)
uint64_t seq_of(const std::string& name) {
    const std::string pre = "conv-", ext = ".keys";
    if (name.size() <= pre.size() + ext.size() || name.compare(0, pre.size(), pre) != 0 ||
        name.compare(name.size() - ext.size(), ext.size(), ext) != 0)
        return 0;
    uint64_t n = 0;
    for (size_t i = pre.size(); i < name.size() - ext.size(); ++i) {
        if (name[i] < '0' || name[i] > '9' || n > (UINT64_MAX - 9) / 10) return 0;
        n = n * 10 + (uint64_t) (name[i] - '0');
    }
    return n;
}
}  // namespace

std::string ConversationDisk::path_of(uint64_t seq, const char* ext) const {
    return (fs::path(dir_) / ("conv-" + std::to_string(seq) + ext)).string();
}

uint64_t ConversationDisk::bytes() const {
    uint64_t n = 0;
    for (const Entry& e : entries_) n += e.file_bytes + e.keys_bytes;
    return n;
}

bool ConversationDisk::open(const std::string& dir, uint64_t cap_bytes, const SessionFileIdentity& id,
                            std::string& error) {
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec || !fs::is_directory(dir, ec)) {
        error = "conversation disk: cannot use " + dir + (ec ? ": " + ec.message() : std::string());
        return false;
    }
    dir_ = dir;
    cap_ = cap_bytes;
    id_ = id;
    entries_.clear();
    next_seq_ = 1;
    for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
        const uint64_t seq = seq_of(it->path().filename().string());
        if (seq == 0 || !it->is_regular_file(ec)) continue;
        next_seq_ = std::max(next_seq_, seq + 1);
        Entry e;
        SessionFileIdentity fid;
        uint64_t recorded = 0;
        if (!read_keys(it->path(), fid, e.cvec, recorded, e.keys) || fid.model != id.model || fid.config != id.config)
            continue;   // another model's entry or an unfinished one: left alone
        const fs::path bin = path_of(seq, ".bin");
        std::error_code fe;
        e.file_bytes = fs::file_size(bin, fe);
        if (fe || e.file_bytes != recorded) continue;
        e.keys_bytes = fs::file_size(it->path(), fe);
        e.seq = seq;
        entries_.push_back(std::move(e));
    }
    if (ec) {
        error = "conversation disk: cannot list " + dir + ": " + ec.message();
        dir_.clear();
        return false;
    }
    std::sort(entries_.begin(), entries_.end(), [](const Entry& a, const Entry& b) { return a.seq < b.seq; });
    enforce_cap();
    return true;
}

bool ConversationDisk::store(SavedConversation&& image, std::string& error, const SessionWriteOptions& options) {
    if (!enabled()) { error = "conversation disk: not open"; return false; }
    if (!image.stage_images.empty()) { error = "conversation disk: a layer split's images are not stored"; return false; }
    image.checkpoints = session_checkpoints_to_save(image.checkpoints);
    if (!image.live.imgs.empty() || std::any_of(image.checkpoints.begin(), image.checkpoints.end(),
                                                [](const ConversationCheckpoint& c) { return !c.imgs.empty(); })) {
        error = "conversation disk: a conversation with pictures is not stored";
        return false;
    }
    const uint64_t seq = next_seq_++;
    const std::string bin = path_of(seq, ".bin"), keys = path_of(seq, ".keys"), tmp = keys + ".tmp";
    size_t written = 0;
    if (!session_file_write(bin, image, id_, written, error, options)) return false;
    std::vector<const ConversationCheckpoint*> ks{&image.live};
    for (const ConversationCheckpoint& c : image.checkpoints) ks.push_back(&c);
    const std::string kb = keys_bytes(id_, image.cvec, (uint64_t) written, ks);
    std::error_code ec;
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out.write(kb.data(), (std::streamsize) kb.size()) || !out.flush()) ec = std::make_error_code(std::errc::io_error);
    }
    if (!ec) fs::rename(tmp, keys, ec);
    if (ec) {
        error = "conversation disk: cannot write " + keys + ": " + ec.message();
        std::error_code ignore;
        fs::remove(tmp, ignore);
        fs::remove(bin, ignore);
        return false;
    }
    Entry e;
    e.seq = seq;
    e.file_bytes = written;
    e.keys_bytes = kb.size();
    e.cvec = image.cvec;
    for (const ConversationCheckpoint* k : ks) {
        ConversationCheckpoint key;
        key.ids = k->ids;
        key.imgs = k->imgs;
        e.keys.push_back(std::move(key));
    }
    entries_.push_back(std::move(e));
    enforce_cap();
    return true;
}

bool ConversationDisk::take(size_t index, SavedConversation& image, const SessionReadLimits& limits,
                            std::string& error) {
    if (index >= entries_.size()) { error = "conversation disk: no such entry"; return false; }
    size_t read = 0;
    SessionStatus st;
    const bool ok = session_file_read(path_of(entries_[index].seq, ".bin"), id_, image, read, error, limits, &st);
    if (!ok && st.error == SessionError::memory) {
        entries_.erase(entries_.begin() + (std::ptrdiff_t) index);   // not now (RAM): its files stay for a later start
        return false;
    }
    remove_entry(index);   // read back (live again) or damaged: either way it leaves the disk
    return ok;
}

void ConversationDisk::remove_entry(size_t index) {
    std::error_code ignore;
    fs::remove(path_of(entries_[index].seq, ".keys"), ignore);   // the keys first: without them the entry is not indexed
    fs::remove(path_of(entries_[index].seq, ".bin"), ignore);
    entries_.erase(entries_.begin() + (std::ptrdiff_t) index);
}

void ConversationDisk::enforce_cap() {
    while (!entries_.empty() && bytes() > cap_) remove_entry(0);
}

}  // namespace strata::core
