// metal/kernels/gemv_il.metal - 0ics: a GEMV for windows of 1..8 tokens whose cost does not grow with the tokens.
//
// y[t][r] = W[r] . x[t], Q8_0 weights repacked in blocks of 32 rows ("IL32"): lane i of a simdgroup owns row 32 b + i,
// the 32 lanes read one unit (32 K values) of their 32 rows as one contiguous 1 KB (two 16-byte loads a lane), and the
// x slice of the unit is the same address for every lane (one broadcast load serves the simdgroup).  The per-lane
// GEMV (gemv.metal) re-reads x per row from L1, so its window costs 1.5x a token's at 4 tokens; here the x traffic is
// per simdgroup, not per row.
//
// Repacked layout (n_out and n_in multiples of 32): quants [n_out / 32][U][32 lanes][32 int8], then the scales
// [n_out / 32][U][32 lanes] half (U = n_in / 32).  Weight = scale * code, as ref_quant's Q8_0.
//
// grid (n_out / 32, G): threadgroup (b, g) covers row block b and the units [g * ups, (g + 1) * ups); its KS
// simdgroups take every KS-th unit and add their partials in simdgroup order (deterministic).  G > 1 writes partial
// sums: y[(g * NT + t) * y_stride + row] (the caller adds the G planes in order).
#include <metal_stdlib>
using namespace metal;

constant int IL_NT [[function_constant(20)]];
// bench probe (function constant 21): 1 = every token reads token 0's x slice (wrong values), 2 = only token 0 does
// the math (wrong values)
constant int IL_PROBE [[function_constant(21)]];
constant int il_probe = is_function_constant_defined(IL_PROBE) ? IL_PROBE : 0;
// 1: gemv_ilx adds the K splits itself (the last threadgroup of a column), y is final
constant int IL_RED [[function_constant(22)]];
constant int il_red = is_function_constant_defined(IL_RED) ? IL_RED : 0;
// gemv_ilx: x staged in chunks of this many units (0 = the threadgroup's whole range)
constant int IL_CH [[function_constant(23)]];
constant int il_ch = is_function_constant_defined(IL_CH) ? IL_CH : 0;

struct IlArgs {
    uint n_in;
    uint n_out;
    uint x_stride;          // floats between tokens' x rows
    uint y_stride;          // floats between tokens' y rows
    uint units_per_split;   // units of one grid.y split
};

static inline float unit_dot(uint4 q0, uint4 q1, device const float4* xv) {
    const uint v[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
    float s0 = 0.0f, s1 = 0.0f;
    for (int j = 0; j < 8; j += 2) {
        s0 += dot(float4(as_type<char4>(v[j])), xv[j]);
        s1 += dot(float4(as_type<char4>(v[j + 1])), xv[j + 1]);
    }
    return s0 + s1;
}

template <int KS, int UN>
kernel void gemv_il_q8(device const uchar* Wq [[buffer(0)]], device const half* Ws [[buffer(1)]],
                       device const float* x [[buffer(2)]], device float* y [[buffer(3)]],
                       constant IlArgs& a [[buffer(4)]], uint3 tg [[threadgroup_position_in_grid]],
                       ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[KS > 1 ? KS * 8 * 32 : 1];
    const int U = int(a.n_in) / 32, b = int(tg.x);
    const int g0 = int(tg.y) * int(a.units_per_split), g1 = min(U, g0 + int(a.units_per_split));
    device const uint4* qb = (device const uint4*)(Wq + ulong(b) * ulong(U) * 1024) + lane * 2;
    device const half* sb = Ws + ulong(b) * ulong(U) * 32 + lane;
    float acc[8];
    for (int t = 0; t < 8; ++t) acc[t] = 0.0f;
    int u = g0 + sg;
    // UN units in flight a lane (their loads issued before the math)
    for (; u + (UN - 1) * KS < g1; u += UN * KS) {
        uint4 q0[UN], q1[UN];
        float d[UN];
        for (int k = 0; k < UN; ++k) {
            const int uk = u + k * KS;
            q0[k] = qb[uk * 64];
            q1[k] = qb[uk * 64 + 1];
            d[k] = float(sb[uk * 32]);
        }
        for (int k = 0; k < UN; ++k) {
            const int uk = u + k * KS;
            for (int t = 0; t < IL_NT; ++t) {
                if (il_probe == 2 && t > 0) continue;
                device const float4* xv = (device const float4*)(x + (il_probe == 1 ? 0 : t) * int(a.x_stride) + uk * 32);
                acc[t] = fma(d[k], unit_dot(q0[k], q1[k], xv), acc[t]);
            }
        }
    }
    for (; u < g1; u += KS) {
        const uint4 q0 = qb[u * 64], q1 = qb[u * 64 + 1];
        const float d = float(sb[u * 32]);
        for (int t = 0; t < IL_NT; ++t) {
            if (il_probe == 2 && t > 0) continue;
            device const float4* xv = (device const float4*)(x + (il_probe == 1 ? 0 : t) * int(a.x_stride) + u * 32);
            acc[t] = fma(d, unit_dot(q0, q1, xv), acc[t]);
        }
    }
    const int row = b * 32 + int(lane);
    device float* yo = y + int(tg.y) * IL_NT * int(a.y_stride);
    if (KS == 1) {
        for (int t = 0; t < IL_NT; ++t) yo[t * int(a.y_stride) + row] = acc[t];
        return;
    }
    for (int t = 0; t < IL_NT; ++t) red[(sg * 8 + t) * 32 + lane] = acc[t];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0)
        for (int t = 0; t < IL_NT; ++t) {
            float v = red[t * 32 + lane];
            for (int k = 1; k < KS; ++k) v += red[(k * 8 + t) * 32 + lane];
            yo[t * int(a.y_stride) + row] = v;
        }
}

#define IL(KS, UN) template [[host_name("gemv_il_q8_" #KS "_" #UN)]] kernel decltype(gemv_il_q8<KS, UN>) gemv_il_q8<KS, UN>;
IL(1, 1) IL(1, 2) IL(1, 4) IL(2, 1) IL(2, 2) IL(2, 4) IL(4, 1) IL(4, 2) IL(4, 4) IL(8, 1) IL(8, 2) IL(8, 4)
IL(16, 1) IL(16, 2)

// RS row blocks a threadgroup, every simdgroup on its own block and all of them over the same units in the same order:
// the x slice one simdgroup loads is in L1 for the others (the KS form above splits the units, so each simdgroup
// streams its own x slices and a window of tokens re-reads x from L2 once per row block).  grid (ceil(RB / RS), G).
template <int RS, int UN>
kernel void gemv_ilr_q8(device const uchar* Wq [[buffer(0)]], device const half* Ws [[buffer(1)]],
                        device const float* x [[buffer(2)]], device float* y [[buffer(3)]],
                        constant IlArgs& a [[buffer(4)]], uint3 tg [[threadgroup_position_in_grid]],
                        ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    const int U = int(a.n_in) / 32, RB = int(a.n_out) / 32;
    const int b = int(tg.x) * RS + sg;
    if (b >= RB) return;
    const int g0 = int(tg.y) * int(a.units_per_split), g1 = min(U, g0 + int(a.units_per_split));
    device const uint4* qb = (device const uint4*)(Wq + ulong(b) * ulong(U) * 1024) + lane * 2;
    device const half* sb = Ws + ulong(b) * ulong(U) * 32 + lane;
    float acc[8];
    for (int t = 0; t < 8; ++t) acc[t] = 0.0f;
    int u = g0;
    for (; u + UN - 1 < g1; u += UN) {
        uint4 q0[UN], q1[UN];
        float d[UN];
        for (int k = 0; k < UN; ++k) {
            q0[k] = qb[(u + k) * 64];
            q1[k] = qb[(u + k) * 64 + 1];
            d[k] = float(sb[(u + k) * 32]);
        }
        for (int k = 0; k < UN; ++k)
            for (int t = 0; t < IL_NT; ++t) {
                if (il_probe == 2 && t > 0) continue;
                device const float4* xv = (device const float4*)(x + (il_probe == 1 ? 0 : t) * int(a.x_stride) + (u + k) * 32);
                acc[t] = fma(d[k], unit_dot(q0[k], q1[k], xv), acc[t]);
            }
    }
    for (; u < g1; ++u) {
        const uint4 q0 = qb[u * 64], q1 = qb[u * 64 + 1];
        const float d = float(sb[u * 32]);
        for (int t = 0; t < IL_NT; ++t) {
            if (il_probe == 2 && t > 0) continue;
            device const float4* xv = (device const float4*)(x + (il_probe == 1 ? 0 : t) * int(a.x_stride) + u * 32);
            acc[t] = fma(d, unit_dot(q0, q1, xv), acc[t]);
        }
    }
    device float* yo = y + int(tg.y) * IL_NT * int(a.y_stride);
    for (int t = 0; t < IL_NT; ++t) yo[t * int(a.y_stride) + b * 32 + int(lane)] = acc[t];
}

#define ILR(RS, UN) template [[host_name("gemv_ilr_q8_" #RS "_" #UN)]] kernel decltype(gemv_ilr_q8<RS, UN>) gemv_ilr_q8<RS, UN>;
ILR(2, 1) ILR(2, 2) ILR(4, 1) ILR(4, 2) ILR(8, 1) ILR(8, 2) ILR(16, 1) ILR(16, 2) ILR(32, 1)

// As gemv_ilr, with the x slices of SL units staged in threadgroup memory by the whole threadgroup (one coalesced load
// a slab), read back as broadcasts; each simdgroup issues its slab's weight loads before the barrier.
static inline float unit_dot_tg(uint4 q0, uint4 q1, threadgroup const float4* xv) {
    const uint v[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
    float s0 = 0.0f, s1 = 0.0f;
    for (int j = 0; j < 8; j += 2) {
        s0 += dot(float4(as_type<char4>(v[j])), xv[j]);
        s1 += dot(float4(as_type<char4>(v[j + 1])), xv[j + 1]);
    }
    return s0 + s1;
}

template <int RS, int SL>
kernel void gemv_ilt_q8(device const uchar* Wq [[buffer(0)]], device const half* Ws [[buffer(1)]],
                        device const float* x [[buffer(2)]], device float* y [[buffer(3)]],
                        constant IlArgs& a [[buffer(4)]], uint3 tg [[threadgroup_position_in_grid]],
                        ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
                        ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float4 xs[8 * SL * 8];   // [token][unit of the slab][8 float4]
    const int U = int(a.n_in) / 32, RB = int(a.n_out) / 32;
    const int b = min(int(tg.x) * RS + int(sg), RB - 1);
    const int g0 = int(tg.y) * int(a.units_per_split), g1 = min(U, g0 + int(a.units_per_split));
    device const uint4* qb = (device const uint4*)(Wq + ulong(b) * ulong(U) * 1024) + lane * 2;
    device const half* sb = Ws + ulong(b) * ulong(U) * 32 + lane;
    device const float4* x4 = (device const float4*)x;
    const int xs4 = int(a.x_stride) / 4;
    float acc[8];
    for (int t = 0; t < 8; ++t) acc[t] = 0.0f;
    for (int u0 = g0; u0 < g1; u0 += SL) {
        const int n = min(SL, g1 - u0);
        uint4 q0[SL], q1[SL];
        float d[SL];
        for (int k = 0; k < SL; ++k)
            if (k < n) {
                q0[k] = qb[(u0 + k) * 64];
                q1[k] = qb[(u0 + k) * 64 + 1];
                d[k] = float(sb[(u0 + k) * 32]);
            }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int e = tid; e < IL_NT * n * 8; e += RS * 32) {
            const int t = e / (n * 8), r = e - t * (n * 8);
            xs[t * SL * 8 + r] = x4[t * xs4 + u0 * 8 + r];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int k = 0; k < SL; ++k)
            if (k < n)
                for (int t = 0; t < IL_NT; ++t) acc[t] = fma(d[k], unit_dot_tg(q0[k], q1[k], xs + (t * SL + k) * 8), acc[t]);
    }
    if (int(tg.x) * RS + int(sg) >= RB) return;
    device float* yo = y + int(tg.y) * IL_NT * int(a.y_stride);
    for (int t = 0; t < IL_NT; ++t) yo[t * int(a.y_stride) + b * 32 + int(lane)] = acc[t];
}

#define ILT(RS, SL) template [[host_name("gemv_ilt_q8_" #RS "_" #SL)]] kernel decltype(gemv_ilt_q8<RS, SL>) gemv_ilt_q8<RS, SL>;
ILT(2, 2) ILT(2, 4) ILT(4, 2) ILT(4, 4) ILT(8, 2) ILT(8, 4) ILT(16, 2) ILT(16, 4) ILT(4, 8) ILT(8, 8)

// As gemv_ilr with NR row blocks a simdgroup (lane i: rows 32 b + i for NR consecutive blocks b): each x float4 a lane
// loads is used by NR rows, so the x load instructions a weight fall NR times.
template <int RS, int NR>
kernel void gemv_ilm_q8(device const uchar* Wq [[buffer(0)]], device const half* Ws [[buffer(1)]],
                        device const float* x [[buffer(2)]], device float* y [[buffer(3)]],
                        constant IlArgs& a [[buffer(4)]], uint3 tg [[threadgroup_position_in_grid]],
                        ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    const int U = int(a.n_in) / 32, RB = int(a.n_out) / 32;
    const int b0 = (int(tg.x) * RS + sg) * NR;
    if (b0 >= RB) return;
    const int g0 = int(tg.y) * int(a.units_per_split), g1 = min(U, g0 + int(a.units_per_split));
    device const uint4* qb[NR];
    device const half* sb[NR];
    for (int i = 0; i < NR; ++i) {
        const int b = min(b0 + i, RB - 1);
        qb[i] = (device const uint4*)(Wq + ulong(b) * ulong(U) * 1024) + lane * 2;
        sb[i] = Ws + ulong(b) * ulong(U) * 32 + lane;
    }
    float acc[NR][8];
    for (int i = 0; i < NR; ++i)
        for (int t = 0; t < 8; ++t) acc[i][t] = 0.0f;
    for (int u = g0; u < g1; ++u) {
        float4 w[NR][8];
        float d[NR];
        for (int i = 0; i < NR; ++i) {
            const uint4 q0 = qb[i][u * 64], q1 = qb[i][u * 64 + 1];
            const uint v[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
            for (int j = 0; j < 8; ++j) w[i][j] = float4(as_type<char4>(v[j]));
            d[i] = float(sb[i][u * 32]);
        }
        for (int t = 0; t < IL_NT; ++t) {
            if (il_probe == 2 && t > 0) continue;
            device const float4* xv = (device const float4*)(x + (il_probe == 1 ? 0 : t) * int(a.x_stride) + u * 32);
            float s0[NR], s1[NR];
            for (int i = 0; i < NR; ++i) s0[i] = s1[i] = 0.0f;
            for (int j = 0; j < 8; j += 2) {
                const float4 xa = xv[j], xb = xv[j + 1];
                for (int i = 0; i < NR; ++i) {
                    s0[i] += dot(w[i][j], xa);
                    s1[i] += dot(w[i][j + 1], xb);
                }
            }
            for (int i = 0; i < NR; ++i) acc[i][t] = fma(d[i], s0[i] + s1[i], acc[i][t]);
        }
    }
    device float* yo = y + int(tg.y) * IL_NT * int(a.y_stride);
    for (int i = 0; i < NR; ++i)
        if (b0 + i < RB)
            for (int t = 0; t < IL_NT; ++t) yo[t * int(a.y_stride) + (b0 + i) * 32 + int(lane)] = acc[i][t];
}

#define ILM(RS, NR) template [[host_name("gemv_ilm_q8_" #RS "_" #NR)]] kernel decltype(gemv_ilm_q8<RS, NR>) gemv_ilm_q8<RS, NR>;
ILM(1, 2) ILM(2, 2) ILM(4, 2) ILM(8, 2) ILM(1, 4) ILM(2, 4) ILM(4, 4) ILM(8, 4) ILM(16, 2)

// As gemv_ilr, with the threadgroup's whole x range (its units, every token) staged in threadgroup memory once, behind
// one barrier (the caller picks G so that NT * units_per_split * 128 bytes fit: tg_mem).
template <int RS, int UN>
kernel void gemv_ilx_q8(device const uchar* Wq [[buffer(0)]], device const half* Ws [[buffer(1)]],
                        device const float* x [[buffer(2)]], device float* y [[buffer(3)]],
                        constant IlArgs& a [[buffer(4)]], device atomic_float* part [[buffer(5)]],
                        coherent(device) device float* cpart [[buffer(7)]],
                        device atomic_uint* cnt [[buffer(6)]], threadgroup float4* xs [[threadgroup(0)]],
                        uint3 tg [[threadgroup_position_in_grid]], uint3 ntg [[threadgroups_per_grid]],
                        ushort sg [[simdgroup_index_in_threadgroup]],
                        ushort lane [[thread_index_in_simdgroup]], ushort tid [[thread_index_in_threadgroup]]) {
    const int U = int(a.n_in) / 32, RB = int(a.n_out) / 32;
    const int b = min(int(tg.x) * RS + int(sg), RB - 1);
    const int ups = int(a.units_per_split);
    const int g0 = int(tg.y) * ups, g1 = min(U, g0 + ups), n = g1 - g0;
    device const uint4* qb = (device const uint4*)(Wq + ulong(b) * ulong(U) * 1024) + lane * 2;
    device const half* sb = Ws + ulong(b) * ulong(U) * 32 + lane;
    // the first units' weights in flight while x is staged
    uint4 p0[UN], p1[UN];
    float pd[UN];
    for (int k = 0; k < UN; ++k) {
        const int uk = min(g0 + k, g1 - 1);
        p0[k] = qb[uk * 64];
        p1[k] = qb[uk * 64 + 1];
        pd[k] = float(sb[uk * 32]);
    }
    device const float4* x4 = (device const float4*)x;
    const int xs4 = int(a.x_stride) / 4;
    // x staged in chunks of CH units (il_ch; 0 = the whole range at once), a barrier a chunk; CH a multiple of UN
    const int CH = il_ch > 0 ? il_ch : ups;
    float acc[8];
    for (int t = 0; t < 8; ++t) acc[t] = 0.0f;
    for (int c0 = 0; c0 < n; c0 += CH) {
        const int cn = min(CH, n - c0);
        if (c0 > 0) threadgroup_barrier(mem_flags::mem_threadgroup);   // the previous chunk is read by everyone
        for (int e = tid; e < IL_NT * cn * 8; e += RS * 32) {
            const int t = e / (cn * 8), r = e - t * (cn * 8);
            xs[t * CH * 8 + r] = x4[t * xs4 + (g0 + c0) * 8 + r];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int u = c0; u < c0 + cn; u += UN) {
            uint4 q0[UN], q1[UN];
            float d[UN];
            for (int k = 0; k < UN; ++k) { q0[k] = p0[k]; q1[k] = p1[k]; d[k] = pd[k]; }
            for (int k = 0; k < UN; ++k) {
                const int uk = min(g0 + u + UN + k, g1 - 1);
                p0[k] = qb[uk * 64];
                p1[k] = qb[uk * 64 + 1];
                pd[k] = float(sb[uk * 32]);
            }
            for (int k = 0; k < UN; ++k)
                if (u + k < c0 + cn)
                    for (int t = 0; t < IL_NT; ++t)
                        acc[t] = fma(d[k], unit_dot_tg(q0[k], q1[k], xs + (t * CH + u + k - c0) * 8), acc[t]);
        }
    }
    const bool live = int(tg.x) * RS + int(sg) < RB;
    const int G = int(ntg.y), row = b * 32 + int(lane);
    if (!il_red || G == 1) {
        if (!live) return;
        device float* yo = y + int(tg.y) * IL_NT * int(a.y_stride);
        for (int t = 0; t < IL_NT; ++t) yo[t * int(a.y_stride) + row] = acc[t];
        return;
    }
    // split K: every threadgroup of the column leaves its partial (atomic stores: the coherent point); the last one to
    // arrive adds the G partials in split order (deterministic) and resets the column's counter for the next call
    // il_red 1: atomic stores and loads; 2: plain accesses through a coherent(device) pointer (MSL 3.2)
    if (live)
        for (int t = 0; t < IL_NT; ++t) {
            const int i = (int(tg.y) * IL_NT + t) * int(a.n_out) + row;
            if (il_red == 2) cpart[i] = acc[t];
            else atomic_store_explicit(&part[i], acc[t], memory_order_relaxed);
        }
    threadgroup_barrier(mem_flags::mem_device);
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    threadgroup uint last;
    if (tid == 0) {
        const uint prev = atomic_fetch_add_explicit(&cnt[tg.x], 1u, memory_order_relaxed);
        last = prev == uint(G - 1) ? 1u : 0u;
        if (last) atomic_store_explicit(&cnt[tg.x], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!last || !live) return;
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    for (int t = 0; t < IL_NT; ++t) {
        float v = 0.0f;
        for (int g = 0; g < G; ++g) {
            const int i = (g * IL_NT + t) * int(a.n_out) + row;
            v += il_red == 2 ? cpart[i] : atomic_load_explicit(&part[i], memory_order_relaxed);
        }
        y[t * int(a.y_stride) + row] = v;
    }
}

#define ILX(RS, UN) template [[host_name("gemv_ilx_q8_" #RS "_" #UN)]] kernel decltype(gemv_ilx_q8<RS, UN>) gemv_ilx_q8<RS, UN>;
ILX(2, 1) ILX(2, 2) ILX(4, 1) ILX(4, 2) ILX(8, 1) ILX(8, 2) ILX(16, 1) ILX(16, 2) ILX(4, 4) ILX(8, 4)
