// metal/kernels/hc_il.metal - 0ics: the hyper-connection read (gr.metal's gr_norm + gr_down + gr_up) in two
// dispatches on IL32 copies of the HC matrices (il_gemv.metal's layout), the same per-element arithmetic:
//
//   hc_dn  grid (ceil(LR / 32 / RS) + 1, G), RS simdgroups.  Threadgroup (x, g) owns the K range [g ups, (g + 1) ups)
//          units, inside one stream c.  It forms rs[t] = rsqrt(mean_d R'[t][c][d]^2 + eps) itself over the whole
//          stream (R' = fma(bo, g_c, R) when apply: gr_norm's), stages xn = (R' * w_norm) * rs for its range in
//          threadgroup memory, and dots it with RS row blocks of w_down (IL32): partial sums loP[g][t][row].  The last
//          column does the 4 F32 inject rows the same way: injP[g][t][c'].  Column 0 of a stream's first split writes
//          rs to rsb[t][c].  R is only read.
//   hc_up  grid (n / 32), 4 simdgroups (one a stream), lane = d.  lo = silu((sum_g loP) / hc) staged per threadgroup;
//          gate[t][c][d] = sigmoid(w_up[c n + d] . lo[t]) (IL32, 10 units); then per (t, d): R'[t][c][d] formed again
//          (and written back in place when apply - every element read and written by this one thread), xn = (R' *
//          w_norm) * rs, mixed = (sum_c xn gate) / hc in c order, written as the prepared planes and per-16 sums
//          (xprep's layout).  Threadgroup 0 also writes inject = sum_g injP (the next read's g).
//
// The sums of squares, the dots and the inject rows add in another order than gr.metal's: the same numbers, other
// roundings.  Splits: ups divides the 80 units of a stream (G a multiple of 4).
#include <metal_stdlib>
using namespace metal;

constant int HC_NT [[function_constant(0)]];   // tokens of the window (1..8)
// bench probe (function constant 1, wrong values): 1 = hc_up adds 1 lo plane, not G; 2 = hc_up skips the mix epilogue;
// 3 = hc_dn skips the sums of squares (rs = 1)
constant int HC_PROBE [[function_constant(1)]];
constant int hc_probe = is_function_constant_defined(HC_PROBE) ? HC_PROBE : 0;

struct HcArgs {
    uint n;        // 2560
    uint hc;       // 4
    uint lr;       // 320
    uint apply;    // fold the previous sublayer's write (bo, inj) into R
    uint G;        // K splits of hc_dn
    uint ups;      // units a split (10240 / 32 / G)
    uint plane;    // floats between the prepared planes of the output (MAXT * n)
    uint maxt;     // token stride of the partial planes
    float eps;
};

static inline float hc_g(device const float* inj, constant HcArgs& a, int t, int c) {
    return a.apply ? 2.0f / (1.0f + exp(-inj[t * int(a.hc) + c] / float(a.hc))) : 0.0f;
}

template <int RS>
kernel void hc_dn(device const uchar* Wq [[buffer(0)]], device const half* Ws [[buffer(1)]],
                  device const float* winj [[buffer(2)]], device const float* R [[buffer(3)]],
                  device const float* bo [[buffer(4)]], device const float* inj [[buffer(5)]],
                  device const float* wn [[buffer(6)]], device float* loP [[buffer(7)]], device float* injP [[buffer(8)]],
                  device float* rsb [[buffer(9)]], constant HcArgs& a [[buffer(10)]],
                  threadgroup float4* xs [[threadgroup(0)]], uint3 tg [[threadgroup_position_in_grid]],
                  uint3 ntg [[threadgroups_per_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                  ushort lane [[thread_index_in_simdgroup]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float red[8][RS];
    threadgroup float rs_tg[8], g_tg[8];
    const int n = int(a.n), hc = int(a.hc), ups = int(a.ups), NIN = hc * n, U = NIN / 32;
    const int u0 = int(tg.y) * ups, c = u0 / (n / 32), d0 = (u0 % (n / 32)) * 32, len4 = ups * 8;
    if (tid < HC_NT) g_tg[tid] = hc_g(inj, a, tid, c);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // rs of stream c, every token: the whole stream's R'
    device const float4* R4 = (device const float4*)R;
    device const float4* bo4 = (device const float4*)bo;
    for (int t = 0; t < HC_NT; ++t) {
        float ss = 0.0f;
        for (int i = tid; i < (hc_probe == 3 ? 0 : n / 4); i += RS * 32) {
            float4 v = R4[((t * hc + c) * n) / 4 + i];
            if (a.apply) v = fma(bo4[(t * n) / 4 + i], float4(g_tg[t]), v);
            ss += dot(v, v);
        }
        ss = simd_sum(ss);
        if (lane == 0) red[t][sg] = ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < HC_NT) {
        float tot = 0.0f;
        for (int k = 0; k < RS; ++k) tot += red[tid][k];
        rs_tg[tid] = rsqrt(tot / float(n) + a.eps);
        if (tg.x == 0 && d0 == 0) rsb[tid * hc + c] = rs_tg[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // xn of the range: (R' * w_norm) * rs
    device const float4* wn4 = (device const float4*)wn;
    for (int e = tid; e < HC_NT * len4; e += RS * 32) {
        const int t = e / len4, i = e - t * len4;
        float4 v = R4[((t * hc + c) * n + d0) / 4 + i];
        if (a.apply) v = fma(bo4[(t * n + d0) / 4 + i], float4(g_tg[t]), v);
        xs[t * len4 + i] = v * wn4[(c * n + d0) / 4 + i] * rs_tg[t];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int ncol = int(ntg.x), RB = int(a.lr) / 32;
    if (int(tg.x) < ncol - 1) {
        const int b = min(int(tg.x) * RS + int(sg), RB - 1);
        device const uint4* qb = (device const uint4*)(Wq + ulong(b) * ulong(U) * 1024) + lane * 2;
        device const half* sb = Ws + ulong(b) * ulong(U) * 32 + lane;
        float acc[8];
        for (int t = 0; t < 8; ++t) acc[t] = 0.0f;
        for (int u = 0; u < ups; ++u) {
            const uint4 q0 = qb[(u0 + u) * 64], q1 = qb[(u0 + u) * 64 + 1];
            const float dsc = float(sb[(u0 + u) * 32]);
            const uint v[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
            for (int t = 0; t < HC_NT; ++t) {
                threadgroup const float4* xv = xs + t * len4 + u * 8;
                float s0 = 0.0f, s1 = 0.0f;
                for (int j = 0; j < 8; j += 2) {
                    s0 += dot(float4(as_type<char4>(v[j])), xv[j]);
                    s1 += dot(float4(as_type<char4>(v[j + 1])), xv[j + 1]);
                }
                acc[t] = fma(dsc, s0 + s1, acc[t]);
            }
        }
        if (int(tg.x) * RS + int(sg) < RB)
            for (int t = 0; t < HC_NT; ++t) loP[(int(tg.y) * int(a.maxt) + t) * int(a.lr) + b * 32 + int(lane)] = acc[t];
    } else {
        // the inject rows (F32 [hc][hc n]): simdgroup s takes rows s, s + RS, ..; lanes split the range
        threadgroup const float* xf = (threadgroup const float*)xs;
        for (int r = sg; r < hc; r += RS) {
            device const float* w = winj + r * NIN + c * n + d0;
            float acc[8];
            for (int t = 0; t < 8; ++t) acc[t] = 0.0f;
            for (int k = lane; k < ups * 32; k += 32) {
                const float wv = w[k];
                for (int t = 0; t < HC_NT; ++t) acc[t] = fma(wv, xf[t * ups * 32 + k], acc[t]);
            }
            for (int t = 0; t < HC_NT; ++t) {
                const float s = simd_sum(acc[t]);
                if (lane == 0) injP[(int(tg.y) * int(a.maxt) + t) * hc + r] = s;
            }
        }
    }
}

kernel void hc_up(device const uchar* Wq [[buffer(0)]], device const half* Ws [[buffer(1)]],
                  device const float* loP [[buffer(2)]], device const float* injP [[buffer(3)]],
                  device float* R [[buffer(4)]], device const float* bo [[buffer(5)]], device const float* inj [[buffer(6)]],
                  device const float* wn [[buffer(7)]], device const float* rsb [[buffer(8)]], device float* xp [[buffer(9)]],
                  device float* inj_out [[buffer(10)]], constant HcArgs& a [[buffer(11)]],
                  uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                  ushort lane [[thread_index_in_simdgroup]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float lo_tg[8][320];
    threadgroup float gate[8][4][32];
    threadgroup float msum[8][32];
    threadgroup float g_tg[8][4];
    const int n = int(a.n), hc = int(a.hc), lr = int(a.lr), G = int(a.G), mt = int(a.maxt);
    // lo: the G partial planes added in split order, 4 values a thread, every plane's load issued before the adds
    device const float4* loP4 = (device const float4*)loP;
    const int lr4 = lr / 4;
    for (int e = tid; e < HC_NT * lr4; e += 128) {
        const int t = e / lr4, k4 = e - t * lr4;
        float4 p[16];
        const int GG = hc_probe == 1 ? 1 : G;
        for (int g = 0; g < 16; ++g)
            if (g < GG) p[g] = loP4[(g * mt + t) * lr4 + k4];
        float4 v = 0.0f;
        for (int g = 0; g < 16; ++g)
            if (g < GG) v += p[g];
        const float4 z = v / float(hc);
        const float4 sl = z / (1.0f + exp(-z));
        for (int j = 0; j < 4; ++j) lo_tg[t][4 * k4 + j] = sl[j];
    }
    if (tid < HC_NT * hc) {
        const int t = tid / hc, c = tid - t * hc;
        g_tg[t][c] = hc_g(inj, a, t, c);
        if (tg.x == 0) {
            float v = 0.0f;
            for (int g = 0; g < G; ++g) v += injP[(g * mt + t) * hc + c];
            inj_out[t * hc + c] = v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // gate: simdgroup c = stream, lane = d (row c n + d = row block c (n / 32) + tg.x, lane)
    {
        const int c = sg, U = lr / 32, b = c * (n / 32) + int(tg.x);
        device const uint4* qb = (device const uint4*)(Wq + ulong(b) * ulong(U) * 1024) + lane * 2;
        device const half* sb = Ws + ulong(b) * ulong(U) * 32 + lane;
        float acc[8];
        for (int t = 0; t < 8; ++t) acc[t] = 0.0f;
        for (int u = 0; u < U; ++u) {
            const uint4 q0 = qb[u * 64], q1 = qb[u * 64 + 1];
            const float dsc = float(sb[u * 32]);
            const uint v[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
            for (int t = 0; t < HC_NT; ++t) {
                threadgroup const float4* xv = (threadgroup const float4*)&lo_tg[t][u * 32];
                float s0 = 0.0f, s1 = 0.0f;
                for (int j = 0; j < 8; j += 2) {
                    s0 += dot(float4(as_type<char4>(v[j])), xv[j]);
                    s1 += dot(float4(as_type<char4>(v[j + 1])), xv[j + 1]);
                }
                acc[t] = fma(dsc, s0 + s1, acc[t]);
            }
        }
        for (int t = 0; t < HC_NT; ++t) gate[t][c][lane] = 1.0f / (1.0f + exp(-acc[t]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint plane = a.plane;
    const float q2s[8] = {1.0f, 0.25f, 0.0625f, 0.015625f, 0.00390625f, 0.0009765625f, 0.000244140625f, 0.00006103515625f};
    const float p4s[4] = {1.0f, 0.00390625f, 1.52587890625e-05f, 5.9604644775390625e-08f};
    for (int it = tid; it < (hc_probe == 2 ? 0 : HC_NT * 32); it += 128) {
        const int t = it / 32, dl = it - t * 32, d = int(tg.x) * 32 + dl, j = d & 15;
        float m = 0.0f;
        for (int c = 0; c < hc; ++c) {
            device float* rp = R + (t * hc + c) * n + d;
            float r = *rp;
            if (a.apply) {
                r = fma(bo[t * n + d], g_tg[t][c], r);
                *rp = r;
            }
            const float xn = r * wn[c * n + d] * rsb[t * hc + c];
            m = fma(xn, gate[t][c][dl], m);
        }
        m = m / float(hc);
        device float* raw = xp + t * n + d;
        raw[0] = m;
        raw[plane] = m * q2s[j & 7];
        raw[2 * plane] = m * p4s[j & 3];
        raw[3 * plane] = (j & 1) ? m * 0.00390625f : m;
        msum[t][dl] = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < HC_NT * 2) {
        const int t = tid / 2, h = tid & 1;
        float s0 = 0.0f, s1 = 0.0f;
        for (int k = 0; k < 16; k += 2) { s0 += msum[t][16 * h + k]; s1 += msum[t][16 * h + k + 1]; }
        xp[4 * plane + t * (n / 16) + int(tg.x) * 2 + h] = s0 + s1;
    }
}

#define HCD(RS) template [[host_name("hc_dn_" #RS)]] kernel decltype(hc_dn<RS>) hc_dn<RS>;
HCD(1) HCD(2) HCD(4) HCD(8)
