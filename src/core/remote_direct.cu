// src/core/remote_direct.cu - see include/strata/core/remote_direct.hpp.
#include "strata/core/remote_direct.hpp"
#include "strata/core/on_device.hpp"
#include "strata/core/remote_experts.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace strata::core {
namespace {
constexpr int64_t H = strata::kernels::cpu::H;
constexpr int MAXT = strata::kernels::cpu::MAXT;
constexpr int CAP = MAXT * 10;
static_assert(CAP <= 128, "remote direct: one plan thread per window entry");

// One token group's plan on the helper: the grouped kernel's layout (distinct held experts in routing order, their
// entries ascending; an entry's output row is its index in the group), and per entry whether the helper owns it and
// its routing weight (copied here once, so the reduction does not read mapped memory per column).
struct DPlan {
    unsigned long long ptr[CAP];
    int32_t start[CAP + 1];
    int32_t dst[CAP];
    int32_t tok[CAP];
    int32_t count[4];
    int32_t own[CAP];
    float w[CAP];
};

bool check(cudaError_t e, const char* what, std::string& err) {
    if (e == cudaSuccess) return true;
    err = std::string("remote direct: ") + what + ": " + cudaGetErrorString(e);
    return false;
}

__global__ void __launch_bounds__(128) rd_plan_kernel(const volatile int32_t* ids, const volatile float* w, int n, int k,
                                                      const unsigned long long* __restrict__ slot, int n_expert,
                                                      DPlan* __restrict__ p, unsigned long long* count) {
    __shared__ int32_t s_id[128];
    __shared__ int s_first[128], s_size[128];
    const int i = threadIdx.x;
    int32_t e = -1;
    unsigned long long ptr = 0;
    if (i < n) {
        e = ids[i];
        ptr = (e >= 0 && e < n_expert) ? slot[e] : 0ull;
        p->own[i] = ptr != 0;
        p->w[i] = w[i];
    }
    const bool held = i < n && ptr != 0;
    s_id[i] = held ? e : -1;
    __syncthreads();
    int first = i, rank = 0, size = 0;
    if (held)
        for (int j = 0; j < n; ++j) {
            const bool same = s_id[j] == e;
            if (same && j < first) first = j;
            if (same && j < i) ++rank;
            size += same;
        }
    s_first[i] = held ? first : -1;
    s_size[i] = held ? size : 0;
    __syncthreads();
    if (held) {
        int group = 0, gstart = 0;
        for (int j = 0; j < first; ++j)
            if (s_first[j] == j) { ++group; gstart += s_size[j]; }
        if (i == first) { p->ptr[group] = ptr; p->start[group] = gstart; }
        p->dst[gstart + rank] = i;
        p->tok[gstart + rank] = i / k;
    }
    if (i == 0) {
        int groups = 0, entries = 0;
        for (int j = 0; j < n; ++j)
            if (s_first[j] == j) { ++groups; entries += s_size[j]; }
        p->start[groups] = entries;
        p->count[0] = groups;
        p->count[1] = entries;
        p->count[2] = 0;
        if (count != nullptr) atomicAdd(count, (unsigned long long) entries);
    }
}

// sum[t][c] = sum over the token's routing slots j (ascending) of the helper's entries: out[t*k+j][c] * w[t*k+j] -
// --remote-expert-opt's reduce_experts, written straight into the mapped host rows.
__global__ void rd_reduce_kernel(const float* __restrict__ out, const DPlan* __restrict__ p, int k, float* sum) {
    const int t = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= H) return;
    float v = 0.0f;
    for (int j = 0; j < k; ++j) {
        const int i = t * k + j;
        if (p->own[i]) v += out[(int64_t) i * H + c] * p->w[i];
    }
    sum[(int64_t) t * H + c] = v;
    __threadfence_system();
}

__global__ void rd_signal_kernel(volatile uint32_t* flag, uint32_t value) {
    __threadfence_system();
    *flag = value;
    __threadfence_system();
}
}  // namespace

RemoteDirect::~RemoteDirect() {
    if (device_ < 0) return;
    {
        const OnDevice on(device_);
        if (stream_) cudaStreamSynchronize(stream_);
        for (auto& g : graphs_) if (g.exec) cudaGraphExecDestroy(g.exec);
        if (d_slot_) cudaFree(d_slot_);
        if (d_plan_) cudaFree(d_plan_);
        if (d_q8_) cudaFree(d_q8_);
        if (d_scratch_) cudaFree(d_scratch_);
        if (d_out_) cudaFree(d_out_);
        if (d_count_) cudaFree(d_count_);
        if (stream_) cudaStreamDestroy(stream_);
    }
    {
        const OnDevice on(primary_);
        if (d_held_) cudaFree(d_held_);
    }
    if (h_flag_) cudaFreeHost(h_flag_);
    if (h_sum_) cudaFreeHost(h_sum_);
}

bool RemoteDirect::open(RemoteExperts& helper, int primary, int64_t n_layers, int64_t n_expert, std::string& err) {
    helper_ = &helper;
    device_ = helper.device_;
    primary_ = primary;
    n_layers_ = n_layers;
    n_expert_ = n_expert;
    if (device_ < 0 || device_ == primary) { err = "remote direct: the helper is not open on its own device"; return false; }
    const size_t table = (size_t) (n_layers * n_expert);
    int64_t ff = strata::kernels::cpu::FF;
    for (const auto& f : strata::kernels::cpu::expert_layout().fmt) ff = std::max<int64_t>(ff, f.n_ff);
    {
        const OnDevice on(primary_);
        if (!check(cudaMalloc((void**) &d_held_, table), "held table", err)) return false;
    }
    // the flag and the reduced rows: written by the helper, read by the primary
    if (!check(cudaHostAlloc((void**) &h_flag_, 64, cudaHostAllocPortable | cudaHostAllocMapped), "flag", err) ||
        !check(cudaHostAlloc((void**) &h_sum_, (size_t) MAXT * H * sizeof(float), cudaHostAllocPortable | cudaHostAllocMapped),
               "sum rows", err))
        return false;
    std::memset(h_flag_, 0, 64);
    std::memset(h_sum_, 0, (size_t) MAXT * H * sizeof(float));
    {
        const OnDevice on(primary_);
        if (!check(cudaHostGetDevicePointer((void**) &m_flag_, h_flag_, 0), "flag on the primary", err) ||
            !check(cudaHostGetDevicePointer((void**) &m_sum_, h_sum_, 0), "sum rows on the primary", err))
            return false;
    }
    const OnDevice on(device_);
    if (!check(cudaHostGetDevicePointer((void**) &z_flag_, h_flag_, 0), "flag on the helper", err) ||
        !check(cudaHostGetDevicePointer((void**) &z_sum_, h_sum_, 0), "sum rows on the helper", err) ||
        !check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "stream", err) ||
        !check(cudaMalloc((void**) &d_slot_, table * sizeof(unsigned long long)), "slot table", err) ||
        !check(cudaMalloc(&d_plan_, 2 * sizeof(DPlan)), "plans", err) ||
        !check(cudaMalloc((void**) &d_q8_, (size_t) 2 * MAXT * (H / 32) * 36), "activations", err) ||
        !check(cudaMalloc(&d_scratch_, strata::kernels::native_expert_scratch_bytes(CAP, ff)), "scratch", err) ||
        !check(cudaMalloc((void**) &d_out_, (size_t) 2 * CAP * H * sizeof(float)), "rows", err) ||
        !check(cudaMalloc((void**) &d_count_, sizeof(unsigned long long)), "counter", err) ||
        !check(cudaMemset(d_count_, 0, sizeof(unsigned long long)), "counter", err) ||
        !check(cudaMemset(d_plan_, 0, 2 * sizeof(DPlan)), "plans", err))
        return false;
    return refresh(err);
}

bool RemoteDirect::refresh(std::string& err) {
    const size_t table = (size_t) (n_layers_ * n_expert_);
    std::vector<unsigned long long> slot(table, 0);
    std::vector<uint8_t> held(table, 0);
    const ExpertCache& c = helper_->cache_;
    for (int64_t l = 0; l < n_layers_; ++l)
        for (int64_t e = 0; e < n_expert_; ++e) {
            const int32_t s = c.slot_of(l, e);
            if (s < 0) continue;
            slot[(size_t) (l * n_expert_ + e)] = (unsigned long long) c.device_slot(s);
            held[(size_t) (l * n_expert_ + e)] = 1;
        }
    {
        const OnDevice on(device_);
        if (!check(cudaMemcpy(d_slot_, slot.data(), table * sizeof(unsigned long long), cudaMemcpyHostToDevice),
                   "slot table upload", err))
            return false;
    }
    const OnDevice on(primary_);
    return check(cudaMemcpy(d_held_, held.data(), table, cudaMemcpyHostToDevice), "held table upload", err);
}

bool RemoteDirect::capture(int T, int G, const int tb[2], const int te[2], int64_t lb, int64_t le, int k, std::string& err) {
    using namespace strata::kernels;
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (!lay.native) { err = "remote direct: needs a native pack"; return false; }
    if (k * MAXT > CAP || k > 10) { err = "remote direct: routing width over 10"; return false; }
    static const int grid = [] { const char* v = std::getenv("STRATA_REMOTE_DIRECT_GRID"); return v ? std::atoi(v) : 0; }();
    if (!check(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal), "begin capture", err)) return false;
    try {
        const cudaStream_t s = stream_;
        const int64_t steps = (le - lb) * G;
        for (int64_t step = 0; step < steps; ++step) {
            const int64_t l = lb + step / G;
            const int grp = (int) (step % G);
            const int n_tok = te[grp] - tb[grp];
            const uint32_t ring = (uint32_t) (step + 1);
            DPlan* p = (DPlan*) d_plan_ + grp;
            uint8_t* q8 = d_q8_ + (size_t) grp * MAXT * (H / 32) * 36;
            float* out = d_out_ + (size_t) grp * CAP * H;
            wait_flag_ge(seq_, ring, s);   // the primary published this group's rows (the host polls the same word)
            rd_plan_kernel<<<1, 128, 0, s>>>((const volatile int32_t*) (ids_ + (size_t) tb[grp] * k),
                                             (const volatile float*) (w_ + (size_t) tb[grp] * k), n_tok * k, k,
                                             d_slot_ + l * n_expert_, (int) n_expert_, p, d_count_);
            quantize_q8_1_rows(x_ + (size_t) tb[grp] * H, n_tok, H, q8, s);
            const auto& f = lay.fmt[(size_t) l];
            const NativeExpertLayout L = native_expert_layout(f.gu_type, f.d_type, f.n_embd, f.n_ff);
            native_expert_grouped(L, p->ptr, p->start, p->count, p->dst, p->tok, CAP, CAP, q8, d_scratch_, out, s, grid);
            rd_reduce_kernel<<<dim3((unsigned) ((H + 255) / 256), (unsigned) n_tok), 256, 0, s>>>(
                out, p, k, z_sum_ + (size_t) tb[grp] * H);
            rd_signal_kernel<<<1, 1, 0, s>>>(z_flag_, ring);
        }
    } catch (const std::exception& e) {
        cudaGraph_t g = nullptr;
        cudaStreamEndCapture(stream_, &g);
        if (g) cudaGraphDestroy(g);
        err = std::string("remote direct: capture: ") + e.what();
        return false;
    }
    cudaGraph_t graph = nullptr;
    const cudaError_t ce = cudaStreamEndCapture(stream_, &graph);
    if (ce != cudaSuccess || graph == nullptr) { err = std::string("remote direct: end capture: ") + cudaGetErrorString(ce); return false; }
    if ((int) graphs_.size() <= T) graphs_.resize((size_t) T + 1);
    Graph& gr = graphs_[(size_t) T];
    if (gr.exec) cudaGraphExecDestroy(gr.exec);
    gr.exec = nullptr;
    const cudaError_t ie = cudaGraphInstantiate(&gr.exec, graph, 0);
    cudaGraphDestroy(graph);
    if (ie != cudaSuccess) { err = std::string("remote direct: instantiate: ") + cudaGetErrorString(ie); return false; }
    gr.G = G; gr.lb = lb; gr.le = le;
    std::fprintf(stderr, "strata remote direct: CUDA%d captured the %d-token window (%d group%s, layers %lld..%lld)\n",
                 device_, T, G, G == 2 ? "s" : "", (long long) lb, (long long) le - 1);
    return true;
}

bool RemoteDirect::launch(int T, int G, const int tb[2], const int te[2], int64_t lb, int64_t le, int k,
                          const float* h_x, const int32_t* h_ids, const float* h_w, const uint32_t* h_seq,
                          std::string& err) {
    if (T < 1 || T > MAXT || G < 1 || G > 2) { err = "remote direct: window out of range"; return false; }
    const OnDevice on(device_);
    if (x_ == nullptr) {   // the verifier's doorbell rows, as the helper sees them (portable mapped memory)
        void *x = nullptr, *ids = nullptr, *w = nullptr, *seq = nullptr;
        if (!check(cudaHostGetDevicePointer(&x, (void*) h_x, 0), "doorbell rows on the helper (allocate them portable)", err) ||
            !check(cudaHostGetDevicePointer(&ids, (void*) h_ids, 0), "doorbell ids on the helper", err) ||
            !check(cudaHostGetDevicePointer(&w, (void*) h_w, 0), "doorbell weights on the helper", err) ||
            !check(cudaHostGetDevicePointer(&seq, (void*) h_seq, 0), "doorbell sequence on the helper", err))
            return false;
        x_ = (const float*) x; ids_ = (const int32_t*) ids; w_ = (const float*) w; seq_ = (const uint32_t*) seq;
        h_seq_ = h_seq;
    }
    const bool have = (int) graphs_.size() > T && graphs_[(size_t) T].exec != nullptr && graphs_[(size_t) T].G == G &&
                      graphs_[(size_t) T].lb == lb && graphs_[(size_t) T].le == le;
    if (!have && !capture(T, G, tb, te, lb, le, k, err)) return false;
    *(volatile uint32_t*) h_flag_ = 0;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (!check(cudaGraphLaunch(graphs_[(size_t) T].exec, stream_), "launch", err)) return false;
    ++windows_;
    return true;
}

bool RemoteDirect::sync(std::string& err) {
    if (device_ < 0 || stream_ == nullptr) return true;
    const OnDevice on(device_);
    return check(cudaStreamSynchronize(stream_), "sync", err);
}

void RemoteDirect::release() {
    if (h_seq_ != nullptr) *(volatile uint32_t*) h_seq_ = 0xffffffffu;
    if (h_flag_ != nullptr) *(volatile uint32_t*) h_flag_ = 0xffffffffu;
    std::atomic_thread_fence(std::memory_order_seq_cst);
}

int64_t RemoteDirect::entries() {
    if (d_count_ == nullptr) return 0;
    const OnDevice on(device_);
    unsigned long long v = 0;
    if (cudaMemcpy(&v, d_count_, sizeof v, cudaMemcpyDeviceToHost) != cudaSuccess) { cudaGetLastError(); return 0; }
    return (int64_t) v;
}
}  // namespace strata::core
