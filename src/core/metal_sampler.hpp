// src/core/metal_sampler.hpp - 0ics: the sampler on the host for the Metal engine (verify.cpp's Metal branch draws a
// request's rows from the window's head logits).  It is sampler.cu's reference sampler_kernel, step for step: the
// logit bias, then the repetition penalties (apply_penalties: count = the token's occurrences in the row's history
// window, the last penalty_last_n slots, -1 = empty), top_k (1..63 as given, else 64) by descending value with ties to
// the lower id (NaN and -inf never kept), top_p over that list in double with min_keep, min_p in logit space, the
// temperature, one Philox 4x32-10 draw keyed by (seed, counter + row) and the inverse-CDF pick (the last kept token
// when the draw runs past the sum).  greedy (no temperature): the argmax of the biased, penalised logits, ties to the
// lower id (sampler_greedy_kernel).  src/core/metal_sampler_test.cpp pins it against sampler_parity.cpp's
// sampled_reference.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

namespace strata::core {

inline float metal_philox_uniform(uint64_t seed, uint64_t counter) {
    uint32_t c0 = (uint32_t) counter, c1 = (uint32_t) (counter >> 32), c2 = (uint32_t) seed, c3 = (uint32_t) (seed >> 32);
    for (int i = 0; i < 10; ++i) {
        const uint64_t m0 = (uint64_t) 0x9E3779B9u * c0, m1 = (uint64_t) 0xBB67AE85u * c2;
        const uint32_t n0 = (uint32_t) (m1 >> 32) ^ c1 ^ (uint32_t) i, n1 = (uint32_t) m1, n2 = (uint32_t) (m0 >> 32) ^ c3,
                       n3 = (uint32_t) m0;
        c0 = n0; c1 = n1; c2 = n2; c3 = n3;
    }
    return (float) (c0 >> 8) * (1.0f / 16777216.0f);
}

// SP: SamplerParams (top_k, top_p, min_p, temperature, min_keep, penalty_repeat / freq / present, seed, counter,
// greedy).  hist: the row's penalty window (hlen slots, most recent last, -1 empty), nullptr / 0 = no penalties.
template <class SP>
int32_t metal_sample_row(const float* l, int nv, const float* bias, const int32_t* hist, int hlen, const SP& p, int row) {
    // the penalised tokens: a token's count in the window, applied to its biased logit (apply_penalties)
    std::vector<std::pair<int, int>> cnt;   // (token, count), sorted by token
    if (hist && hlen > 0) {
        for (int i = 0; i < hlen; ++i)
            if (hist[i] >= 0 && hist[i] < nv) cnt.push_back({hist[i], 1});
        std::sort(cnt.begin(), cnt.end());
        size_t w = 0;
        for (size_t i = 0; i < cnt.size(); ++i) {
            if (w > 0 && cnt[w - 1].first == cnt[i].first) ++cnt[w - 1].second;
            else cnt[w++] = cnt[i];
        }
        cnt.resize(w);
    }
    std::vector<float> pen(cnt.size());
    for (size_t i = 0; i < cnt.size(); ++i) {
        const int v = cnt[i].first, c = cnt[i].second;
        float x = bias ? l[v] + bias[v] : l[v];
        if (x <= 0.0f) x *= p.penalty_repeat;
        else x /= p.penalty_repeat;
        x -= (float) c * p.penalty_freq + 1.0f * p.penalty_present;
        pen[i] = x;
    }
    size_t ci = 0;   // the scans below run v upward: the next penalised token
    auto val = [&](int v) {
        while (ci < cnt.size() && cnt[ci].first < v) ++ci;
        if (ci < cnt.size() && cnt[ci].first == v) return pen[ci];
        return bias ? l[v] + bias[v] : l[v];
    };
    // (value, id) ranks before (value', id') when value > value', or equal values and id < id'
    auto better = [](float a, int ia, float b, int ib) { return a > b || (a == b && ia < ib); };
    if (p.greedy) {
        float bv = -INFINITY;
        int bi = nv;
        for (int v = 0; v < nv; ++v) {
            const float x = val(v);
            if (x > bv || (x == bv && v < bi)) { bv = x; bi = v; }
        }
        return bi < nv ? bi : 0;
    }
    constexpr int KMAX = 64;
    int k = (p.top_k > 0 && p.top_k < KMAX) ? p.top_k : KMAX;
    if (k > nv) k = nv;
    // the k best in a heap with the worst on top, then in descending order
    std::vector<std::pair<float, int>> h;
    h.reserve((size_t) k + 1);
    auto worse = [&](const std::pair<float, int>& a, const std::pair<float, int>& b) { return better(a.first, a.second, b.first, b.second); };
    for (int v = 0; v < nv; ++v) {
        const float x = val(v);
        if (!(x == x) || x == -INFINITY) continue;
        if ((int) h.size() < k) {
            h.push_back({x, v});
            std::push_heap(h.begin(), h.end(), worse);
        } else if (better(x, v, h.front().first, h.front().second)) {
            std::pop_heap(h.begin(), h.end(), worse);
            h.back() = {x, v};
            std::push_heap(h.begin(), h.end(), worse);
        }
    }
    std::sort(h.begin(), h.end(), [&](const auto& a, const auto& b) { return better(a.first, a.second, b.first, b.second); });
    if (h.empty()) return 0;
    k = (int) h.size();
    int n_keep = k;
    if (p.top_p < 1.0f) {
        double sum = 0.0;
        for (int i = 0; i < k; ++i) sum += std::exp((double) h[(size_t) i].first - (double) h[0].first);
        double cum = 0.0;
        int cut = k;
        for (int i = 0; i < k; ++i) {
            cum += std::exp((double) h[(size_t) i].first - (double) h[0].first) / sum;
            if (cum >= (double) p.top_p) { cut = i + 1; break; }
        }
        if (cut < p.min_keep) cut = p.min_keep < k ? p.min_keep : k;
        n_keep = cut;
    }
    if (p.min_p > 0.0f) {
        const float thresh = h[0].first + std::log(p.min_p);
        for (int i = 0; i < n_keep; ++i)
            if (h[(size_t) i].first < thresh) { n_keep = i; break; }
    }
    if (n_keep < 1) n_keep = 1;
    const float inv_t = p.temperature > 0.0f ? 1.0f / p.temperature : 0.0f;
    auto scaled = [&](int i) { return h[(size_t) i].first * inv_t; };
    float smx = scaled(0);
    for (int i = 1; i < n_keep; ++i) smx = std::fmax(smx, scaled(i));
    double sum = 0.0;
    for (int i = 0; i < n_keep; ++i) sum += std::exp((double) scaled(i) - (double) smx);
    int pick = h[(size_t) n_keep - 1].second;
    const float u = metal_philox_uniform(p.seed, p.counter + (uint64_t) row);
    double cum = 0.0;
    for (int i = 0; i < n_keep; ++i) {
        cum += std::exp((double) scaled(i) - (double) smx) / sum;
        if ((double) u < cum) { pick = h[(size_t) i].second; break; }
    }
    return pick;
}

}  // namespace strata::core
