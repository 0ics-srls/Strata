// metal/kernels/il_gemv.metal - 0ics: the decode window's GEMV on IL32 copies of the dense Q8_0 projections
// (gemv_il.metal is the bench of the forms; this is its ilx form, the fastest at 4-8 tokens):
//
//   y[t][r] = W[r] . x[t] for 1..8 tokens, W the rows of up to three matrices with the same input concatenated
//   (attn_qkv + attn_gate, attn_q + attn_k + attn_v, ffn_gate_shexp + ffn_up_shexp): one dispatch, its rows sent to
//   three outputs.
//
// Why: the per-lane GEMV (gemv.metal) re-reads x per row, so a window of 4 tokens costs 1.5x a token's (2560 -> 10240:
// 41 us against 27 on the M5 Ultra); the window's cost is its x traffic, not its weights.  Here lane i of a simdgroup
// owns row 32 b + i: the 32 lanes read one unit (32 K values) of their 32 rows as one contiguous 1 KB (two 16-byte
// loads a lane), and every lane reads the same x slice, staged once per threadgroup in threadgroup memory.  A
// threadgroup's RS simdgroups are RS row blocks over the same units.
//
// IL32 layout (n_out, n_in multiples of 32, U = n_in / 32): quants [n_out / 32][U][32 lanes][32 int8], scales
// [n_out / 32][U][32 lanes] half.  Weight = scale * code (ref_quant's Q8_0); a lane's unit dot is 8 float4 dots
// with fp32 x, scaled by the block's scale and added to its row's sum unit by unit (the order of ref_quant's dot
// differs: same numbers, other rounding).
//
// grid (ceil(n_out / 32 / RS), G): threadgroup (c, g) covers row blocks c RS .. + RS and the units [g ups, (g + 1) ups).
// G = 1: y is final.  G > 1: each split writes its partial sums to its own plane (pl* floats apart) and the reader
// adds the G planes in order.  The x range of a threadgroup is staged in chunks of CH units (0: all at once; tg_mem =
// NT * (CH or ups) * 128 bytes), one barrier a chunk; CH is a multiple of UN.
#include <metal_stdlib>
using namespace metal;

constant int ILS_NT [[function_constant(0)]];   // tokens of the window (1..8)
constant int ILS_CH [[function_constant(1)]];   // units a staged x chunk (0: the threadgroup's whole range)
constant int ils_ch = is_function_constant_defined(ILS_CH) ? ILS_CH : 0;

struct IlsArgs {
    uint n_in;
    uint n_out;      // rows of the concatenated matrix
    uint x_stride;   // floats between tokens' x rows
    uint ups;        // units a grid.y split
    uint end0;       // rows [0, end0) go to y0, [end0, end1) to y1, [end1, n_out) to y2 (multiples of 32)
    uint end1;
    uint ys0, ys1, ys2;   // floats between tokens' rows of each output
    uint pl0, pl1, pl2;   // floats between grid.y planes of each output (G > 1)
};

static inline float il_unit_dot(uint4 q0, uint4 q1, threadgroup const float4* xv) {
    const uint v[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
    float s0 = 0.0f, s1 = 0.0f;
    for (int j = 0; j < 8; j += 2) {
        s0 += dot(float4(as_type<char4>(v[j])), xv[j]);
        s1 += dot(float4(as_type<char4>(v[j + 1])), xv[j + 1]);
    }
    return s0 + s1;
}

template <int RS, int UN>
kernel void ils_q8(device const uchar* Wq [[buffer(0)]], device const half* Ws [[buffer(1)]],
                   device const float* x [[buffer(2)]], device float* y0 [[buffer(3)]], device float* y1 [[buffer(4)]],
                   device float* y2 [[buffer(5)]], constant IlsArgs& a [[buffer(6)]],
                   threadgroup float4* xs [[threadgroup(0)]], uint3 tg [[threadgroup_position_in_grid]],
                   ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
                   ushort tid [[thread_index_in_threadgroup]]) {
    const int U = int(a.n_in) / 32, RB = int(a.n_out) / 32;
    const int b = min(int(tg.x) * RS + int(sg), RB - 1);
    const int ups = int(a.ups);
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
    const int CH = ils_ch > 0 ? ils_ch : ups;
    float acc[8];
    for (int t = 0; t < 8; ++t) acc[t] = 0.0f;
    for (int c0 = 0; c0 < n; c0 += CH) {
        const int cn = min(CH, n - c0);
        if (c0 > 0) threadgroup_barrier(mem_flags::mem_threadgroup);   // the previous chunk is read by everyone
        for (int e = tid; e < ILS_NT * cn * 8; e += RS * 32) {
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
                    for (int t = 0; t < ILS_NT; ++t)
                        acc[t] = fma(d[k], il_unit_dot(q0[k], q1[k], xs + (t * CH + u + k - c0) * 8), acc[t]);
        }
    }
    if (int(tg.x) * RS + int(sg) >= RB) return;
    const int row = b * 32 + int(lane);
    device float* y;
    int r, ys;
    if (row < int(a.end0)) { y = y0 + int(tg.y) * int(a.pl0); r = row; ys = int(a.ys0); }
    else if (row < int(a.end1)) { y = y1 + int(tg.y) * int(a.pl1); r = row - int(a.end0); ys = int(a.ys1); }
    else { y = y2 + int(tg.y) * int(a.pl2); r = row - int(a.end1); ys = int(a.ys2); }
    for (int t = 0; t < ILS_NT; ++t) y[t * ys + r] = acc[t];
}

#define ILS(RS, UN) template [[host_name("ils_q8_" #RS "_" #UN)]] kernel decltype(ils_q8<RS, UN>) ils_q8<RS, UN>;
ILS(2, 1) ILS(2, 2) ILS(4, 1) ILS(4, 2) ILS(8, 1) ILS(8, 2) ILS(16, 1)
