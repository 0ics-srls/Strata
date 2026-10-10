#pragma once
// STRATA_REMOTE_DIRECT=1 (opt-in, with --remote-expert-opt and one helper GPU): the helper serves the verify window
// without the host.  Today each layer goes GPU -> host -> helper -> host -> GPU: the pool thread sees the doorbell,
// plans, stages and launches the helper's kernels, waits for them, adds the result and raises the flag the primary
// spins on.  Here the helper runs a captured graph per window size that waits on the primary's doorbell itself
// (the same mapped sequence the host polls), plans its own entries on the device from its residency table, computes
// them, reduces them with the routing weights straight into mapped host rows and raises its own flag; the primary
// waits on that flag only where it adds the sum.  The primary plans its own resident entries on the device (E-6)
// and treats the helper's as done, so a layer whose experts are all in VRAM (primary or helper) needs nothing from
// the host.  No P2P: everything crosses through pinned host memory, as before.
//
// The arithmetic is the --remote-expert-opt path's: the same q8_1 activation, the same grouped kernel per entry,
// the same reduction order (routing slot j ascending, weight applied per entry), the same add into the combine.

#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {
class RemoteExperts;

class RemoteDirect {
public:
    RemoteDirect() = default;
    ~RemoteDirect();
    RemoteDirect(const RemoteDirect&) = delete;
    RemoteDirect& operator=(const RemoteDirect&) = delete;

    /// After the helper's cache is open.  `primary` is the device the verify window runs on.
    bool open(RemoteExperts& helper, int primary, int64_t n_layers, int64_t n_expert, std::string& err);
    /// Rebuild both residency tables from the helper's cache (after its tier changed); never with a window in flight.
    bool refresh(std::string& err);

    /// On the primary: 1 where the helper holds (layer, expert), [n_layers * n_expert].
    const uint8_t* held_primary() const { return d_held_; }
    /// The helper's flag and its reduced rows (MAXT x n_embd), as the primary sees them (mapped host memory).
    const uint32_t* flag_primary() const { return m_flag_; }
    const float* sum_primary() const { return m_sum_; }

    /// One window: reset the flag, then launch the helper's graph for T tokens in G groups ([tb, te) each) over
    /// layers [lb, le).  The doorbell rows are the verifier's mapped host buffers (host pointers: x (T, n_embd),
    /// ids / weights (T, k), seq).  The graph of a T is captured on its first launch.
    bool launch(int T, int G, const int tb[2], const int te[2], int64_t lb, int64_t le, int k, const float* h_x,
                const int32_t* h_ids, const float* h_w, const uint32_t* h_seq, std::string& err);
    /// On the primary's stream (captured into its window): wait for the helper's flag to reach `ring`, then
    /// dst[0, n*n_embd) += the helper's reduced rows of tokens [tb, tb+n).  One launch.
    void wait_add(float* dst, int tb, int n, uint32_t ring, void* stream) const;
    /// Wait for the last launched window on the helper.
    bool sync(std::string& err);
    /// #267: raise the helper's own wait past any ring (the doorbell sequence it spins on), so a window the primary
    /// abandoned drains.
    void release();

    int device() const { return device_; }
    /// Entries the helper computed, windows launched (cumulative; entries read back from the device).
    int64_t entries();
    int64_t windows() const { return windows_; }
    /// STRATA_REMOTE_DIRECT_PROFILE=1: the helper's graph stamps its stages (its own clock); the mean per window since
    /// the last call, as one line (empty when off).
    std::string profile_report();

private:
    bool capture(int T, int G, const int tb[2], const int te[2], int64_t lb, int64_t le, int k, std::string& err);

    RemoteExperts* helper_ = nullptr;
    int device_ = -1, primary_ = -1;
    int64_t n_layers_ = 0, n_expert_ = 0;
    cudaStream_t stream_ = nullptr;
    unsigned long long* d_slot_ = nullptr;   ///< helper: [n_layers * n_expert] blob address or 0
    uint8_t* d_held_ = nullptr;              ///< primary: [n_layers * n_expert]
    void* d_plan_ = nullptr;                 ///< helper: one plan per token group (two at most)
    uint8_t* d_q8_ = nullptr;                ///< helper: q8_1 rows, one block of MAXT per group
    void* d_scratch_ = nullptr;
    float* d_out_ = nullptr;                 ///< helper: one row per window entry (MAXT * 10)
    unsigned long long* d_count_ = nullptr;  ///< helper: entries computed
    uint32_t* h_flag_ = nullptr; uint32_t* m_flag_ = nullptr; uint32_t* z_flag_ = nullptr;   // host / primary / helper
    float* h_sum_ = nullptr; float* m_sum_ = nullptr; float* z_sum_ = nullptr;
    const float* x_ = nullptr; const int32_t* ids_ = nullptr; const float* w_ = nullptr; const uint32_t* seq_ = nullptr;
    const uint32_t* h_seq_ = nullptr;
    struct Graph { cudaGraphExec_t exec = nullptr; int G = 0; int64_t lb = 0, le = 0; };
    std::vector<Graph> graphs_;
    int64_t windows_ = 0;
    static constexpr int kStamps = 5;        ///< per step: before the plan (+ wait), after it, quantize, experts, reduce+flag
    unsigned* d_ticket_ = nullptr;           ///< helper: blocks of the reduce that finished (the last raises the flag)
    bool prof_on_ = false;
    unsigned long long* d_stamp_ = nullptr;  ///< helper: steps x kStamps
    std::vector<unsigned long long> h_stamp_;
    int64_t last_steps_ = 0;
    double prof_sum_[kStamps] = {};
    int64_t prof_windows_ = 0;
};
}  // namespace strata::core
