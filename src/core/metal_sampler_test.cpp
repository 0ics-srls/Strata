// src/core/metal_sampler_test.cpp - 0ics: metal_sampler.hpp (the Metal engine's host sampler) against
// sampler_parity.cpp's sampled_reference (the host form of sampler.cu's sampler_kernel, with its Philox and its
// penalties), copied here verbatim without the Gumbel branch.  Random rows (with deliberate ties), random penalty
// windows (repeats, empty slots), every combination of top_k, top_p, min_p, temperature, penalties, seeds: the picks
// must be identical; the greedy pick against the penalised argmax; the Philox stream against the transcription.
//   c++ -std=c++17 -O2 -I src/core src/core/metal_sampler_test.cpp -o /tmp/metal_sampler_test && /tmp/metal_sampler_test
#include "metal_sampler.hpp"

#include <cstdio>
#include <random>
#include <vector>

namespace {
struct SP {
    int top_k = 20;
    float top_p = 0.95f;
    float min_p = 0.0f;
    float temperature = 1.0f;
    int min_keep = 1;
    float penalty_repeat = 1.0f;
    float penalty_freq = 0.0f;
    float penalty_present = 0.0f;
    uint64_t seed = 0;
    uint64_t counter = 0;
    bool greedy = false;
};

// ---- sampler_parity.cpp, verbatim (PhiloxRound, host_philox_uniform, sampled_reference without the Gumbel branch)
struct PhiloxRound {
    uint32_t& c0; uint32_t& c1; uint32_t& c2; uint32_t& c3;
    void step(uint32_t k0, uint32_t k1) const {
        const uint32_t hi0 = (uint32_t) (((uint64_t) 0x9E3779B9u * c0) >> 32);
        const uint32_t hi1 = (uint32_t) (((uint64_t) 0xBB67AE85u * c2) >> 32);
        const uint32_t lo0 = 0x9E3779B9u * c0;
        const uint32_t lo1 = 0xBB67AE85u * c2;
        const uint32_t n0 = hi1 ^ c1 ^ k0;
        const uint32_t n1 = lo1;
        const uint32_t n2 = hi0 ^ c3 ^ k1;
        const uint32_t n3 = lo0;
        c0 = n0; c1 = n1; c2 = n2; c3 = n3;
    }
};
float host_philox_uniform(uint64_t seed, uint64_t counter) {
    uint32_t c0 = (uint32_t) counter, c1 = (uint32_t) (counter >> 32);
    uint32_t c2 = (uint32_t) seed, c3 = (uint32_t) (seed >> 32);
    PhiloxRound r{c0, c1, c2, c3};
    for (int i = 0; i < 10; ++i) r.step((uint32_t) i, 0u);
    return (float) (c0 >> 8) * (1.0f / 16777216.0f);
}
int sampled_reference(const std::vector<float>& l, const std::vector<int>& hist, const SP& p, int row) {
    auto penal = [&](float logit, int count) {
        if (count <= 0) return logit;
        if (logit <= 0.0f) logit *= p.penalty_repeat; else logit /= p.penalty_repeat;
        logit -= (float) count * p.penalty_freq + (count > 0 ? 1.0f : 0.0f) * p.penalty_present;
        return logit;
    };
    auto count = [&](int v) { int c = 0; for (int h : hist) if (h == v) ++c; return c; };
    const int nv = (int) l.size();
    const int KMAX = 64;
    const int k = std::min(nv, (p.top_k > 0 && p.top_k < KMAX) ? p.top_k : KMAX);
    std::vector<int> sel_ids;
    std::vector<float> sel_logit;
    std::vector<char> taken((size_t) nv, 0);
    for (int i = 0; i < k; ++i) {
        int best = -1; float bv = 0;
        for (int v = 0; v < nv; ++v) {
            if (taken[(size_t) v]) continue;
            const float s = penal(l[(size_t) v], count(v));
            if (best < 0 || s > bv) { best = v; bv = s; }
        }
        taken[(size_t) best] = 1;
        sel_ids.push_back(best); sel_logit.push_back(bv);
    }
    const int n_sel = (int) sel_ids.size();
    int n_keep = n_sel;
    if (p.top_p < 1.0f) {
        double sum = 0.0;
        for (int i = 0; i < n_sel; ++i) sum += std::exp((double) sel_logit[(size_t) i] - (double) sel_logit[0]);
        double cum = 0.0;
        int cut = n_sel;
        for (int i = 0; i < n_sel; ++i) {
            cum += std::exp((double) sel_logit[(size_t) i] - (double) sel_logit[0]) / sum;
            if (cum >= (double) p.top_p) { cut = i + 1; break; }
        }
        if (cut < p.min_keep) cut = p.min_keep < n_sel ? p.min_keep : n_sel;
        n_keep = cut;
    }
    if (p.min_p > 0.0f) {
        const float thresh = sel_logit[0] + std::log(p.min_p);
        for (int i = 0; i < n_keep; ++i)
            if (sel_logit[(size_t) i] < thresh) { n_keep = i; break; }
    }
    const float inv_t = p.temperature > 0.0f ? 1.0f / p.temperature : 0.0f;
    auto scaled = [&](int i) { return sel_logit[(size_t) i] * inv_t; };
    float smx = scaled(0);
    for (int i = 1; i < n_keep; ++i) smx = std::fmax(smx, scaled(i));
    double sum = 0.0;
    for (int i = 0; i < n_keep; ++i) sum += std::exp((double) scaled(i) - (double) smx);
    int pick = sel_ids[(size_t) (n_keep - 1)];
    const float u = host_philox_uniform(p.seed, p.counter + (uint64_t) row);
    double cum = 0.0;
    for (int i = 0; i < n_keep; ++i) {
        cum += std::exp((double) scaled(i) - (double) smx) / sum;
        if ((double) u < cum) { pick = sel_ids[(size_t) i]; break; }
    }
    return pick;
}
// the greedy kernel's pick: the penalised argmax, ties to the lower id
int greedy_reference(const std::vector<float>& l, const std::vector<int>& hist, const SP& p) {
    SP q = p;
    q.top_k = 1; q.top_p = 1.0f; q.min_p = 0.0f; q.temperature = 1.0f;
    return sampled_reference(l, hist, q, 0);
}
}  // namespace

int main() {
    std::mt19937_64 rng(42);
    long n = 0, bad = 0;
    for (uint64_t c = 0; c < 100000; c += 7)
        for (uint64_t seed : {0ull, 1ull, 0xDEADBEEFCAFEull})
            if (strata::core::metal_philox_uniform(seed, c) != host_philox_uniform(seed, c)) ++bad;
    std::printf("philox: %s\n", bad ? "DIFFERS" : "identical");
    const int NV = 6000, HL = 64;
    std::uniform_int_distribution<int> tok(0, NV - 1), small(0, 40);
    for (int rowset = 0; rowset < 40; ++rowset) {
        std::vector<float> l((size_t) NV);
        std::normal_distribution<float> nd(0.0f, rowset % 3 == 0 ? 1.0f : 4.0f);
        for (auto& x : l) x = nd(rng);
        if (rowset % 2) for (auto& x : l) x = std::round(x * 4.0f) / 4.0f;   // many ties
        // a window of 64 slots: some empty (-1), repeats, and some of the top tokens so the penalties bite
        std::vector<int32_t> win((size_t) HL);
        for (int i = 0; i < HL; ++i) win[(size_t) i] = i < 6 ? -1 : (i % 3 == 0 ? small(rng) : tok(rng));
        for (int i = 0; i < 8; ++i) win[(size_t) (10 + i)] = (int) (std::max_element(l.begin(), l.end()) - l.begin());
        std::vector<int> hvec;
        for (int32_t x : win) if (x >= 0) hvec.push_back(x);
        for (int pen = 0; pen < 3; ++pen) {
            SP p;
            if (pen == 1) { p.penalty_repeat = 1.1f; p.penalty_present = 0.5f; }
            if (pen == 2) { p.penalty_repeat = 1.0f; p.penalty_freq = 0.3f; p.penalty_present = 1.5f; }
            const std::vector<int> none;
            const std::vector<int>& hh = pen ? hvec : none;
            const int32_t* hp = pen ? win.data() : nullptr;
            const int hn = pen ? HL : 0;
            // greedy
            {
                SP g = p;
                g.greedy = true;
                ++n;
                if (strata::core::metal_sample_row(l.data(), NV, nullptr, hp, hn, g, 0) != greedy_reference(l, hh, p)) ++bad;
            }
            for (int tk : {0, 1, 5, 20, 64, 100})
                for (float tp : {1.0f, 0.95f, 0.8f})
                    for (float mp : {0.0f, 0.05f})
                        for (float tt : {0.3f, 0.6f, 1.0f, 1.7f})
                            for (uint64_t seed : {7ull, 123456789ull}) {
                                SP q = p;
                                q.top_k = tk; q.top_p = tp; q.min_p = mp; q.temperature = tt; q.seed = seed;
                                q.counter = (uint64_t) rowset * 1000;
                                for (int row = 0; row < 3; ++row) {
                                    ++n;
                                    if (strata::core::metal_sample_row(l.data(), NV, nullptr, hp, hn, q, row) != sampled_reference(l, hh, q, row)) ++bad;
                                }
                            }
        }
    }
    std::printf("%ld picks, %ld differ from the reference: %s\n", n, bad, bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
