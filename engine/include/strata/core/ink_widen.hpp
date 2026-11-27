// strata-inkling: a BF16 / F16 mtp.safetensors (the original drafter checkpoint) keeps the drafter's norm weights, biases and fuse_weight
// narrow. The CUDA drafter reads those as F32 (the Q8_0 GGUF path widens them while loading), so a narrow copy made it read garbage:
// 0 percent of the drafted tokens were accepted (RTX 3090, UD-Q4_K_XL + the BF16 mtp.safetensors, 2026-11-23).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace strata::core {

// Weight format codes as the inkling loaders use them: 101 = F32, 102 = BF16, 103 = F16.
inline bool ink_dspark_wants_f32(const std::string& name, size_t rank, int wfmt) {
    if (wfmt != 102 && wfmt != 103) return false;
    return rank <= 1 || name.find("norm") != std::string::npos || name.find("bias") != std::string::npos ||
           name.find("fuse_weight") != std::string::npos;
}

// 0ics: the official HF checkpoint (thinkingmachines/Inkling-2-Mini, NVFP4/MXFP8 experts) keeps the trunk's norms, short-conv
// weights, loop controllers, l2-clip scales, rel_logits_proj and the router in BF16, while the CUDA runner reads every trunk tensor
// that is not a GEMM/GEMV operand as F32 (the .strata and GGUF packs store them so; the CPU backend widens all of them): read as
// F32, BF16 pairs gave garbage text on sm_70 and sm_89 (2026-11-27, ubuntu-ai: CPU coherent, CUDA garbage, experts on the CPU
// still garbage). The trunk tensors the runner passes to linear() / the embedding lookup keep their 16-bit format.
// STRATA_INK_HF_F32=0 keeps every trunk tensor as stored (the upstream behaviour, for A/B runs).
inline bool ink_trunk_wants_f32(const std::string& name, int wfmt) {
    if (wfmt != 102 && wfmt != 103) return false;
    static const bool off = [] { const char* e = std::getenv("STRATA_INK_HF_F32"); return e && e[0] == '0'; }();
    if (off) return false;
    static const char* const kGemm[] = {
        ".attn.wq_du.weight", ".attn.wr_du.weight", ".attn.gate_attn.weight", ".attn.wk_dv.weight", ".attn.wv_dv.weight",
        ".attn.attn_sink_proj.weight", ".attn.wo_ud.weight", ".attn.k_sconv.weight_proj", ".attn.v_sconv.weight_proj",
        "model.llm.embed.weight", "model.llm.unembed.weight"};
    for (const char* s : kGemm) {
        const size_t n = std::strlen(s);
        if (name.size() >= n && name.compare(name.size() - n, n, s) == 0) return false;
    }
    return true;
}

// 0ics STRATA_INK_DENSE_Q8=A-B: the attention GEMM matrices (q, r, gate, k, v, sink, o, the dynamic-conv projections) of the
// layers A..B are stored as Q8_ROW (one f32 scale per output row + int8, the format the runners already read: the q8row GEMV
// with int8 activations per 64 for M <= 16, dequantized tiles for a prompt chunk). The dense BF16 trunk is ~3.5 GB read per
// token against ~1.8 GB of experts: half of it is the largest lever left on an RTX 4090 stage. Layers outside A..B (a V100 stage:
// dp4a is emulated on sm_70) keep BF16. Unset = upstream. A malformed range ends the load with its reason.
struct InkDenseQ8Range {
    int first = -1;
    int last = -1;
    bool on() const noexcept { return first >= 0 && last >= first; }
};

inline InkDenseQ8Range ink_dense_q8_range() {
    static const InkDenseQ8Range r = [] {
        InkDenseQ8Range out;
        const char* e = std::getenv("STRATA_INK_DENSE_Q8");
        if (!e || !*e) return out;
        int a = -1, b = -1;
        if (std::sscanf(e, "%d-%d", &a, &b) != 2 || a < 0 || b < a || b > 23) {
            std::fprintf(stderr, "STRATA_INK_DENSE_Q8 must be A-B with 0 <= A <= B <= 23 (got '%s')\n", e);
            std::exit(2);
        }
        out.first = a;
        out.last = b;
        return out;
    }();
    return r;
}

inline bool ink_trunk_wants_q8row(const std::string& name, int wfmt, size_t rank) {
    if ((wfmt != 102 && wfmt != 103) || rank != 2) return false;
    const InkDenseQ8Range r = ink_dense_q8_range();
    if (!r.on()) return false;
    static const char* const kPrefix = "model.llm.layers.";
    if (name.rfind(kPrefix, 0) != 0) return false;
    const int layer = std::atoi(name.c_str() + std::strlen(kPrefix));
    if (layer < r.first || layer > r.last) return false;
    static const char* const kAttnGemm[] = {
        ".attn.wq_du.weight", ".attn.wr_du.weight", ".attn.gate_attn.weight", ".attn.wk_dv.weight", ".attn.wv_dv.weight",
        ".attn.attn_sink_proj.weight", ".attn.wo_ud.weight", ".attn.k_sconv.weight_proj", ".attn.v_sconv.weight_proj"};
    for (const char* s : kAttnGemm) {
        const size_t n = std::strlen(s);
        if (name.size() >= n && name.compare(name.size() - n, n, s) == 0) return true;
    }
    return false;
}

inline void ink_widen16_to_f32(const uint8_t* src, int wfmt, size_t n, uint8_t* dst);

// rows x cols 16-bit values (wfmt 102 BF16 / 103 F16) -> Q8_ROW at dst: [rows] f32 scales (absmax / 127), then [rows, cols]
// int8 rounded to nearest; an all-zero row has scale 0.
inline void ink_q8row_from16(const uint8_t* src, int wfmt, size_t rows, size_t cols, uint8_t* dst) {
    float* scales = reinterpret_cast<float*>(dst);
    int8_t* q = reinterpret_cast<int8_t*>(dst + rows * sizeof(float));
    std::vector<float> row(cols);
    for (size_t r = 0; r < rows; ++r) {
        ink_widen16_to_f32(src + r * cols * 2, wfmt, cols, reinterpret_cast<uint8_t*>(row.data()));
        float amax = 0.0f;
        for (float v : row) amax = std::max(amax, std::fabs(v));
        const float sc = amax / 127.0f;
        const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
        scales[r] = sc;
        for (size_t c = 0; c < cols; ++c) {
            const long v = std::lround(row[c] * inv);
            q[r * cols + c] = static_cast<int8_t>(v > 127 ? 127 : (v < -127 ? -127 : v));
        }
    }
}

// n little-endian 16-bit values (BF16 when wfmt == 102, IEEE half when 103) widened to n floats at dst.
inline void ink_widen16_to_f32(const uint8_t* src, int wfmt, size_t n, uint8_t* dst) {
    for (size_t i = 0; i < n; ++i) {
        const uint32_t h = static_cast<uint32_t>(src[2 * i]) | (static_cast<uint32_t>(src[2 * i + 1]) << 8);
        uint32_t u;
        if (wfmt == 102) {
            u = h << 16;
        } else {
            const uint32_t sign = (h & 0x8000u) << 16;
            uint32_t exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu;
            if (exp == 0) {
                if (man == 0) {
                    u = sign;
                } else {
                    exp = 127 - 15 + 1;
                    while (!(man & 0x400u)) { man <<= 1; --exp; }
                    man &= 0x3FFu;
                    u = sign | (exp << 23) | (man << 13);
                }
            } else if (exp == 31) {
                u = sign | 0x7F800000u | (man << 13);
            } else {
                u = sign | ((exp + 127 - 15) << 23) | (man << 13);
            }
        }
        std::memcpy(dst + 4 * i, &u, 4);
    }
}

}  // namespace strata::core
