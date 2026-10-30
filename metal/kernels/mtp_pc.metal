// metal/kernels/mtp_pc.metal - 0ics: the drafter's prompt rows as one batch (prompt_chain.inc's mtp_prefill_pc)
//
//   mtp_combine_rows  R[t][c][d] = h[t][c][d] + e[t][d]: msl/mtp.metal's mtp_combine with h in the row layout the
//                     batched fc_hidden GEMM leaves (row t * hc + c), the same addition
#include <metal_stdlib>
using namespace metal;

struct McArgs { uint n, hc, n_tok, pad; };

// ple_hist_shift: the PLE history after a prompt chunk of n_tok rows, on the GPU (prompt_chain.inc's host loop, the same
// copies): per column c of the 10240, the 9 rows kept are the last of [old history | the chunk's normalized rows].
// One thread a column.
kernel void ple_hist_shift(device const float* nw [[buffer(0)]], device float* hist [[buffer(1)]], constant uint4& a [[buffer(2)]],
                           uint c [[thread_position_in_grid]]) {
    const int nT = int(a.x), H = int(a.y), DIM = int(a.z);
    if (int(c) >= DIM) return;
    device float* col = hist + c * uint(H);
    float tmp[32];
    int k = 0;
    const int keep_old = max(0, H - nT);
    for (int r = H - keep_old; r < H; ++r) tmp[k++] = col[r];
    for (int t = max(0, nT - H); t < nT; ++t) tmp[k++] = nw[t * DIM + int(c)];
    for (int r = 0; r < H; ++r) col[r] = tmp[r];
}

kernel void mtp_combine_rows(device const float* h [[buffer(0)]], device const float* e [[buffer(1)]], device float* R [[buffer(2)]],
                             constant McArgs& a [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
    const uint per = a.hc * a.n;
    if (gid >= a.n_tok * per) return;
    const uint t = gid / per, d = gid % a.n;
    R[gid] = h[gid] + e[t * a.n + d];
}
