// metal/kernels/mtp_pc.metal - 0ics: the drafter's prompt rows as one batch (prompt_chain.inc's mtp_prefill_pc)
//
//   mtp_combine_rows  R[t][c][d] = h[t][c][d] + e[t][d]: msl/mtp.metal's mtp_combine with h in the row layout the
//                     batched fc_hidden GEMM leaves (row t * hc + c), the same addition
#include <metal_stdlib>
using namespace metal;

struct McArgs { uint n, hc, n_tok, pad; };

kernel void mtp_combine_rows(device const float* h [[buffer(0)]], device const float* e [[buffer(1)]], device float* R [[buffer(2)]],
                             constant McArgs& a [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
    const uint per = a.hc * a.n;
    if (gid >= a.n_tok * per) return;
    const uint t = gid / per, d = gid % a.n;
    R[gid] = h[gid] + e[t * a.n + d];
}
