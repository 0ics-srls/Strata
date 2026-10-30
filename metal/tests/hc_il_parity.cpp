// metal/tests/hc_il_parity.cpp - 0ics: the two-dispatch hyper-connection read on IL32 weights (hc_il.metal: hc_dn +
// hc_up) against the engine's three dispatches on the same Q8_0 weights (gr.metal: gr_norm + gr_down_q8f + gr_up_q8,
// the Unsloth form: Q8_0 down / up, F32 inject rows), and the speed of both.
//
//   hc_il_parity                     parity, 1 / 4 / 8 tokens, with and without the folded write
//   hc_il_parity --bench [--ntok N]  GPU time of one read (48 weight sets rotated, 96 reads in a chain)
#include "mtl_rt.h"
#include "ref_quant.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {
constexpr int N = 2560, HC = 4, LR = 320, NIN = HC * N, MAXT = 8;
struct GrArgs { uint32_t n_embd, hc, hc_lr, n_tok, apply; float eps; };
struct HcArgs { uint32_t n, hc, lr, apply, G, ups, plane, maxt; float eps; };
double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
struct Err {
    double se = 0, sr = 0, mx = 0, mr = 0;
    void add(double g, double r) { se += (g - r) * (g - r); sr += r * r; mx = std::max(mx, std::fabs(g - r)); mr = std::max(mr, std::fabs(r)); }
    double l2() const { return sr > 0 ? std::sqrt(se / sr) : std::sqrt(se); }
};
// rows x n_in Q8_0 (ggml rows) -> IL32: quants [rows / 32][U][32][32], then scales [rows / 32][U][32] half
void il32(const uint8_t* q8, int rows, int n_in, uint8_t* q, uint8_t* s) {
    const int U = n_in / 32;
    for (int r = 0; r < rows; ++r)
        for (int u = 0; u < U; ++u) {
            const uint8_t* blk = q8 + ((size_t)r * U + u) * 34;
            const size_t i = ((size_t)(r / 32) * U + u) * 32 + r % 32;
            std::memcpy(q + i * 32, blk + 2, 32);
            std::memcpy(s + i * 2, blk, 2);
        }
}
}  // namespace

int main(int argc, char** argv) {
    bool bench = false;
    std::vector<int> ntoks = {1, 4, 8};
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--bench") bench = true;
        else if (a == "--ntok") ntoks = {std::atoi(argv[++i])};
    }
    mtl::Context cx;
    const int lib = cx.compile({"quant.metal", "units.metal", "gr.metal"});
    const int lil = cx.compile({"hc_il.metal"}, "#define NO_GGML_COMMON 1\n");
    std::printf("device %s\n", cx.device_name().c_str());
    const int SETS = bench ? 48 : 1;
    // one weight set: Q8_0 down [LR][NIN], F32 inject [HC][NIN], Q8_0 up [NIN][LR], F32 w_norm [NIN], and the IL32 copies
    const size_t qd = (size_t)LR * NIN / 32 * 34, bi = (size_t)HC * NIN * 4, qu = (size_t)NIN * LR / 32 * 34, bn = (size_t)NIN * 4;
    const size_t ild_q = (size_t)LR * NIN, ild_s = (size_t)LR * NIN / 32 * 2, ilu_q = (size_t)NIN * LR, ilu_s = (size_t)NIN * LR / 32 * 2;
    const size_t o_d = 0, o_i = qd, o_u = o_i + bi, o_n = o_u + qu, o_dq = o_n + bn, o_ds = o_dq + ild_q, o_uq = o_ds + ild_s,
                 o_us = o_uq + ilu_q, set = (o_us + ilu_s + 255) / 256 * 256;
    mtl::Buffer W = cx.alloc(set * SETS);
    {
        uint8_t* b = W.as<uint8_t>();
        ref::Rng r(3);
        std::vector<float> f((size_t)LR * NIN);
        for (auto& v : f) v = r.sym() * 0.02f;
        ref::quantize_q8_0(f.data(), f.size(), b + o_d);
        float* wi = (float*)(b + o_i);
        for (int i = 0; i < HC * NIN; ++i) wi[i] = r.sym() * 0.02f;
        for (auto& v : f) v = r.sym() * 0.1f;
        ref::quantize_q8_0(f.data(), f.size(), b + o_u);
        float* wn = (float*)(b + o_n);
        for (int i = 0; i < NIN; ++i) wn[i] = 1.0f + 0.1f * r.sym();
        il32(b + o_d, LR, NIN, b + o_dq, b + o_ds);
        il32(b + o_u, NIN, LR, b + o_uq, b + o_us);
        for (int s = 1; s < SETS; ++s) std::memcpy(b + s * set, b, set);
    }
    const size_t xpb = 4 * (4 * (size_t)MAXT * N + (size_t)MAXT * N / 16);
    mtl::Buffer R0 = cx.alloc(4 * MAXT * NIN), RA = cx.alloc(4 * MAXT * NIN), RB = cx.alloc(4 * MAXT * NIN),
                bo = cx.alloc(4 * MAXT * N), inj = cx.alloc(4 * MAXT * HC), xn = cx.alloc(4 * MAXT * NIN),
                lo = cx.alloc(4 * MAXT * LR), injA = cx.alloc(4 * MAXT * HC), injB = cx.alloc(4 * MAXT * HC),
                xpA = cx.alloc(xpb), xpB = cx.alloc(xpb), loP = cx.alloc(4 * 32 * MAXT * LR), injP = cx.alloc(4 * 32 * MAXT * HC),
                rsb = cx.alloc(4 * MAXT * HC);
    {
        ref::Rng r(9);
        for (int i = 0; i < MAXT * NIN; ++i) R0.as<float>()[i] = r.sym() * 2.0f;
        for (int i = 0; i < MAXT * N; ++i) bo.as<float>()[i] = r.sym();
        for (int i = 0; i < MAXT * HC; ++i) inj.as<float>()[i] = r.sym() * 3.0f;
    }
    int fails = 0;
    for (int nt : ntoks) {
        mtl::Pipeline pn = cx.pipeline(lib, "gr_norm"), pd = cx.pipeline(lib, "gr_down_q8f_8_8", mtl::Constants().i(1, nt)),
                      pu = cx.pipeline(lib, "gr_up_q8", mtl::Constants().i(1, nt));
        const int G = std::getenv("HC_G") ? std::atoi(std::getenv("HC_G")) : (nt <= 4 ? 8 : 16), ups = NIN / 32 / G;
        const int RS = std::getenv("HC_RS") ? std::atoi(std::getenv("HC_RS")) : 2;
        mtl::Pipeline hd = cx.pipeline(lil, "hc_dn_" + std::to_string(RS), mtl::Constants().i(0, nt).i(1, std::getenv("HC_PROBE") ? std::atoi(std::getenv("HC_PROBE")) : 0)),
                      hu = cx.pipeline(lil, "hc_up", mtl::Constants().i(0, nt).i(1, std::getenv("HC_PROBE") ? std::atoi(std::getenv("HC_PROBE")) : 0));
        for (int apply = 0; apply < 2; ++apply) {
            if (bench && !apply) continue;
            GrArgs a{N, HC, LR, (uint32_t)nt, (uint32_t)apply, 1e-6f}, au = a;
            au.n_tok = MAXT;
            HcArgs h{N, HC, LR, (uint32_t)apply, (uint32_t)G, (uint32_t)ups, (uint32_t)(MAXT * N), MAXT, 1e-6f};
            auto old_read = [&](size_t o, std::vector<mtl::Dispatch>& ds) {
                mtl::Dispatch d1, d2, d3;
                d1.p = &pn;
                d1.args = {mtl::Arg::B(RA), mtl::Arg::B(bo), mtl::Arg::B(inj), mtl::Arg::B(W, o + o_n), mtl::Arg::B(xn), mtl::Arg::V(a)};
                d1.grid[0] = HC;
                d1.grid[1] = nt;
                d1.tg[0] = 256;
                d2.p = &pd;
                d2.args = {mtl::Arg::B(W, o + o_d), mtl::Arg::B(W, o + o_i), mtl::Arg::B(xn), mtl::Arg::B(lo), mtl::Arg::B(injA),
                           mtl::Arg::V(a)};
                d2.grid[0] = LR + HC;
                d2.tg[0] = 256;
                d3.p = &pu;
                d3.args = {mtl::Arg::B(W, o + o_u), mtl::Arg::B(lo), mtl::Arg::B(xn), mtl::Arg::B(xpA), mtl::Arg::V(au)};
                d3.grid[0] = N / 16;
                d3.tg[0] = 256;
                ds.push_back(d1);
                ds.push_back(d2);
                ds.push_back(d3);
            };
            auto new_read = [&](size_t o, std::vector<mtl::Dispatch>& ds) {
                mtl::Dispatch d1, d2;
                d1.p = &hd;
                d1.args = {mtl::Arg::B(W, o + o_dq), mtl::Arg::B(W, o + o_ds), mtl::Arg::B(W, o + o_i), mtl::Arg::B(RB), mtl::Arg::B(bo),
                           mtl::Arg::B(inj), mtl::Arg::B(W, o + o_n), mtl::Arg::B(loP), mtl::Arg::B(injP), mtl::Arg::B(rsb),
                           mtl::Arg::V(h)};
                d1.grid[0] = (LR / 32 + RS - 1) / RS + 1;
                d1.grid[1] = G;
                d1.tg[0] = 32 * RS;
                d1.tg_mem = (uint32_t)(nt * ups * 128);
                d2.p = &hu;
                d2.args = {mtl::Arg::B(W, o + o_uq), mtl::Arg::B(W, o + o_us), mtl::Arg::B(loP), mtl::Arg::B(injP), mtl::Arg::B(RB),
                           mtl::Arg::B(bo), mtl::Arg::B(inj), mtl::Arg::B(W, o + o_n), mtl::Arg::B(rsb), mtl::Arg::B(xpB),
                           mtl::Arg::B(injB), mtl::Arg::V(h)};
                d2.grid[0] = N / 32;
                d2.tg[0] = 128;
                // HC_ONLY=dn / up: one of the two (timing only)
                const char* only = std::getenv("HC_ONLY");
                if (!only || std::string(only) == "dn") ds.push_back(d1);
                if (!only || std::string(only) == "up") ds.push_back(d2);
            };
            if (bench) {
                for (int which = 0; which < 2; ++which) {
                    std::vector<mtl::Dispatch> ds;
                    for (int k = 0; k < 96; ++k) {
                        if (which) new_read((size_t)(k % SETS) * set, ds);
                        else old_read((size_t)(k % SETS) * set, ds);
                    }
                    // R is folded in place 96 times: keep it bounded (apply with a small bo would still grow; restart each run)
                    std::memcpy(RA.ptr, R0.ptr, 4 * MAXT * NIN);
                    std::memcpy(RB.ptr, R0.ptr, 4 * MAXT * NIN);
                    cx.warm(ds, 0.1);
                    std::vector<double> t;
                    for (int i = 0; i < 7; ++i) t.push_back(cx.run(ds) / 96);
                    std::printf("bench ntok %d %s: %7.2f us a read\n", nt, which ? "hc_il (2 dispatches)" : "gr.metal (3)  ", median(t) * 1e6);
                }
                continue;
            }
            std::memcpy(RA.ptr, R0.ptr, 4 * MAXT * NIN);
            std::memcpy(RB.ptr, R0.ptr, 4 * MAXT * NIN);
            std::memset(xpA.ptr, 0, xpb);
            std::memset(xpB.ptr, 0, xpb);
            std::vector<mtl::Dispatch> da, db;
            old_read(0, da);
            new_read(0, db);
            cx.run(da);
            cx.run(db);
            Err eR, eM, eI, eS;
            bool rbit = true;
            for (int t = 0; t < nt; ++t) {
                for (int i = 0; i < NIN; ++i) {
                    const float x = RA.as<float>()[t * NIN + i], y = RB.as<float>()[t * NIN + i];
                    eR.add(y, x);
                    rbit = rbit && std::memcmp(&x, &y, 4) == 0;
                }
                for (int k = 0; k < 4; ++k)
                    for (int d = 0; d < N; ++d) eM.add(xpB.as<float>()[(size_t)k * MAXT * N + t * N + d], xpA.as<float>()[(size_t)k * MAXT * N + t * N + d]);
                for (int i = 0; i < N / 16; ++i) eS.add(xpB.as<float>()[4 * MAXT * N + t * (N / 16) + i], xpA.as<float>()[4 * MAXT * N + t * (N / 16) + i]);
                for (int c = 0; c < HC; ++c) eI.add(injB.as<float>()[t * HC + c], injA.as<float>()[t * HC + c]);
            }
            const bool ok = rbit && eM.l2() < 1e-5 && eS.l2() < 1e-5 && eI.l2() < 1e-5;
            fails += !ok;
            std::printf("parity ntok %d apply %d (G %d, RS %d): R' %s, planes %.2e (max %.2e), sums16 %.2e, inject %.2e  %s\n", nt,
                        apply, G, RS, rbit ? "bitwise" : "DIFFERENT", eM.l2(), eM.mx / std::max(eM.mr, 1e-30), eS.l2(), eI.l2(),
                        ok ? "PASS" : "FAIL");
        }
    }
    if (!bench) std::printf("%s\n", fails ? "SOME FAILED" : "ALL PASS");
    return fails ? 1 : 0;
}
