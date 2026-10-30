// metal/tests/gemv_il_bench.cpp - 0ics: the IL32 GEMV (gemv_il.metal) against double, and its speed per tile on the
// model's dense shapes, 1..8 tokens (the same timing as gemv_parity --bench: dependent dispatches over weight copies
// that do not fit the SLC, median of 5).
//
//   gemv_il_bench                      the Flash-Next dense Q8_0 shapes, 1 and 4 tokens
//   gemv_il_bench --one NIN NOUT [--ntok N]   one shape, every tile printed
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
struct IlArgs { uint32_t n_in, n_out, x_stride, y_stride, units_per_split; };
struct Shape { const char* name; int n_in, n_out; };
double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
}  // namespace

int main(int argc, char** argv) {
    std::vector<Shape> shapes = {{"attn_qkv", 2560, 10240}, {"attn_gate", 2560, 6144}, {"ssm_out", 6144, 2560},
                                 {"attn_q", 2560, 12288},   {"attn_out", 6144, 2560}, {"hc_down", 10240, 320},
                                 {"hc_up", 320, 10240},     {"shexp_gu", 2560, 640},  {"shexp_dn", 640, 2560},
                                 {"attn_kv", 2560, 512}};
    std::vector<int> ntoks = {1, 4};
    bool one = false;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        if (s == "--one") { shapes = {{"one", std::atoi(argv[i + 1]), std::atoi(argv[i + 2])}}; one = true; i += 2; }
        else if (s == "--ntok") ntoks = {std::atoi(argv[++i])};
    }
    mtl::Context cx;
    const int lib = cx.compile({"gemv_il.metal"}, "#define NO_GGML_COMMON 1\n");
    std::printf("device %s\n%-10s %6s %6s %4s %9s %8s %7s  tile (KS|-RS,UN,G)  max rel err\n", cx.device_name().c_str(), "shape",
                "n_in", "n_out", "ntok", "us", "GB/s", "MB");
    constexpr int MAXT = 8;
    for (const Shape& s : shapes) {
        const int U = s.n_in / 32, RB = s.n_out / 32;
        const size_t qbytes = (size_t)s.n_out * s.n_in, sbytes = (size_t)s.n_out * U * 2, mb = qbytes + sbytes;
        const int copies = std::max<int>(1, (int)((768u << 20) / mb) + 1);
        // the standard Q8_0 rows, then the IL32 repack of copy 0, replicated
        std::vector<uint8_t> std8((size_t)s.n_out * U * 34);
        {
            ref::Rng r(7);
            std::vector<float> row(s.n_in);
            for (int o = 0; o < s.n_out; ++o) {
                for (int k = 0; k < s.n_in; ++k) row[k] = r.sym() * 0.05f;
                ref::quantize_q8_0(row.data(), s.n_in, std8.data() + (size_t)o * U * 34);
            }
        }
        mtl::Buffer W = cx.alloc(mb * copies);
        {
            uint8_t* q = W.as<uint8_t>();
            uint16_t* sc = (uint16_t*)(q + qbytes);
            for (int b = 0; b < RB; ++b)
                for (int u = 0; u < U; ++u)
                    for (int l = 0; l < 32; ++l) {
                        const uint8_t* blk = std8.data() + ((size_t)(b * 32 + l) * U + u) * 34;
                        std::memcpy(q + (((size_t)b * U + u) * 32 + l) * 32, blk + 2, 32);
                        std::memcpy(sc + ((size_t)b * U + u) * 32 + l, blk, 2);
                    }
            for (int c = 1; c < copies; ++c) std::memcpy(q + (size_t)c * mb, q, mb);
        }
        mtl::Buffer x = cx.alloc(4 * (size_t)MAXT * s.n_in), y = cx.alloc(4 * (size_t)32 * MAXT * s.n_out);
        // gemv_ilx's split-K reduction (IL_RED=1): the partials and one counter a threadgroup column
        const int red = std::getenv("IL_RED") ? std::atoi(std::getenv("IL_RED")) : 0;
        const int stress = std::getenv("IL_STRESS") ? std::atoi(std::getenv("IL_STRESS")) : 0;
        mtl::Buffer part = cx.alloc(4 * (size_t)32 * MAXT * s.n_out), cnt = cx.alloc(4 * (size_t)RB),
                    ys = cx.alloc(4 * (size_t)64 * MAXT * s.n_out);
        std::memset(cnt.ptr, 0, 4 * (size_t)RB);
        {
            ref::Rng r(5);
            for (int i = 0; i < MAXT * s.n_in; ++i) x.as<float>()[i] = r.sym();
        }
        // the double reference of copy 0
        std::vector<double> yr((size_t)MAXT * s.n_out);
        {
            std::vector<float> wf(s.n_in);
            for (int o = 0; o < s.n_out; ++o) {
                ref::dequant(ref::Q8_0, std8.data() + (size_t)o * U * 34, wf.data(), s.n_in);
                for (int t = 0; t < MAXT; ++t) {
                    double a = 0;
                    for (int k = 0; k < s.n_in; ++k) a += (double)wf[k] * x.as<float>()[t * s.n_in + k];
                    yr[(size_t)t * s.n_out + o] = a;
                }
            }
        }
        for (int nt : ntoks) {
            double best = 1e9, best_err = 0;
            int bks = 0, bun = 0, bg = 0, bfam = 0, bch = 0;
            // fam 0: gemv_il_q8_KS_UN (KS simdgroups split the units of one row block); fam 1: gemv_ilr_q8_RS_UN (RS row
            // blocks a threadgroup over the same units) - `ks` is KS or RS
            // fam 2: gemv_ilt_q8_RS_SL (as fam 1, x slabs of SL units in threadgroup memory) - `un` is SL
            for (int fam : {0, 1, 2, 3, 4})
            for (int ks : {1, 2, 4, 8, 16, 32})
                for (int un : {1, 2, 4, 8})
                    for (int G : {1, 2, 4, 8, 16, 32})
                    for (int ch : {0, 8, 16, 32, 64}) {
                        if (fam != 4 && ch) continue;
                        if (std::getenv("IL_FAM") && fam != std::atoi(std::getenv("IL_FAM"))) continue;
                        if (std::getenv("IL_GMAX") && G > std::atoi(std::getenv("IL_GMAX"))) continue;
                        if (fam == 0 && (ks == 32 || un == 8 || (ks == 16 && un == 4))) continue;
                        if (fam == 1 && (ks == 1 || un >= 4 || (ks == 32 && un == 2))) continue;
                        if (fam == 2 && (ks == 1 || ks == 32 || un == 1 || (un == 8 && ks != 4 && ks != 8))) continue;
                        // fam 3: gemv_ilm_q8_RS_NR (NR row blocks a simdgroup) - `un` is NR
                        if (fam == 3 && (ks > 16 || (un != 2 && un != 4) || (ks == 16 && un != 2))) continue;
                        // fam 4: gemv_ilx_q8_RS_UN (the threadgroup's x range staged once)
                        if (fam == 4 && (ks == 1 || ks == 32 || un == 8 || (un == 4 && ks != 4 && ks != 8))) continue;
                        if (fam == 1 && ks > RB) continue;
                        if (fam == 3 && ks * un > RB) continue;
                        const int ups = (U + G - 1) / G;
                        if (ups < (fam ? 1 : ks) * (fam == 3 ? 1 : un) && !(G == 1)) continue;
                        if (fam == 2 && ups < un) continue;
                        if (fam == 4 && (size_t)nt * (ch ? ch : ups) * 128 > 32768) continue;
                        if (fam == 4 && ch && (ch >= ups || ch % un)) continue;
                        if (G > 1 && RB >= 160 * (fam == 3 ? ks * un : fam ? ks : 1) && fam != 4) continue;   // enough threadgroups: no K split across them
                        const std::string kn = std::string(fam == 4 ? "gemv_ilx_q8_" : fam == 3 ? "gemv_ilm_q8_" : fam == 2 ? "gemv_ilt_q8_" : fam ? "gemv_ilr_q8_" : "gemv_il_q8_") + std::to_string(ks) + "_" + std::to_string(un);
                        mtl::Pipeline p = cx.pipeline(lib, kn, mtl::Constants().i(20, nt).i(21, std::getenv("IL_PROBE") ? std::atoi(std::getenv("IL_PROBE")) : 0).i(22, red).i(23, ch));
                        const bool reduced = red && fam == 4 && G > 1;
                        IlArgs a{(uint32_t)s.n_in, (uint32_t)s.n_out, (uint32_t)s.n_in, (uint32_t)s.n_out, (uint32_t)ups};
                        auto disp = [&](int c, const mtl::Buffer* yb = nullptr, size_t yoff = 0) {
                            mtl::Dispatch d;
                            d.p = &p;
                            d.args = {mtl::Arg::B(W, (size_t)c * mb), mtl::Arg::B(W, (size_t)c * mb + qbytes), mtl::Arg::B(x),
                                      mtl::Arg::B(yb ? *yb : y, yoff), mtl::Arg::V(a), mtl::Arg::B(part), mtl::Arg::B(cnt),
                                      mtl::Arg::B(part)};
                            d.grid[0] = fam == 4 ? (RB + ks - 1) / ks : fam == 3 ? (RB + ks * un - 1) / (ks * un) : fam ? (RB + ks - 1) / ks : RB;
                            d.grid[1] = G;
                            d.tg[0] = 32 * ks;
                            if (fam == 4) d.tg_mem = (uint32_t)(nt * (ch ? ch : ups) * 128);
                            return d;
                        };
                        cx.run({disp(0)});
                        double err = 0, mag = 0;
                        for (int t = 0; t < nt; ++t)
                            for (int o = 0; o < s.n_out; ++o) {
                                double v = 0;
                                for (int g = 0; g < (reduced ? 1 : G); ++g) v += y.as<float>()[((size_t)g * nt + t) * s.n_out + o];
                                err = std::max(err, std::fabs(v - yr[(size_t)t * s.n_out + o]));
                                mag = std::max(mag, std::fabs(yr[(size_t)t * s.n_out + o]));
                            }
                        if (reduced && stress) {
                            // the expected y: the partials the run left, added on the host in split order (float)
                            std::vector<float> ex((size_t)nt * s.n_out);
                            for (int t = 0; t < nt; ++t)
                                for (int o = 0; o < s.n_out; ++o) {
                                    float v = 0.0f;
                                    for (int g = 0; g < G; ++g) v += part.as<float>()[((size_t)g * nt + t) * s.n_out + o];
                                    ex[(size_t)t * s.n_out + o] = v;
                                }
                            long bad = 0, runs = 0;
                            for (int it = 0; it < stress; ++it) {
                                std::vector<mtl::Dispatch> sd;
                                for (int k = 0; k < 64; ++k) sd.push_back(disp(k % copies, &ys, (size_t)k * nt * s.n_out * 4));
                                cx.run(sd);
                                for (int k = 0; k < 64; ++k, ++runs)
                                    if (std::memcmp(ys.as<float>() + (size_t)k * nt * s.n_out, ex.data(), 4 * ex.size())) ++bad;
                            }
                            bool first = std::memcmp(y.as<float>(), ex.data(), 4 * ex.size()) == 0;
                            std::printf("   stress ilx %d,%d,%2d: %ld runs, %ld not bitwise the split-order sum (first run %s)\n", ks, un, G,
                                        runs, bad, first ? "equal" : "DIFFERENT");
                        }
                        std::vector<mtl::Dispatch> ds;
                        const int R = std::max(copies, 16);
                        for (int k = 0; k < R; ++k) ds.push_back(disp(k % copies));
                        cx.warm(ds, 0.05);
                        std::vector<double> tt;
                        for (int i = 0; i < 5; ++i) tt.push_back(cx.run(ds) / R);
                        const double m = median(tt);
                        if (one) std::printf("   %s ch %2d tile %2d,%d,%2d: %8.2f us %7.1f GB/s  err %.1e\n", fam == 4 ? "ilx" : fam == 3 ? "ilm" : fam == 2 ? "ilt" : fam ? "ilr" : "il ", ch, ks, un, G, m * 1e6, mb / m / 1e9, err / mag);
                        if (m < best) { best = m; best_err = err / mag; bks = fam ? -ks : ks; bfam = fam; bun = un; bg = G; bch = ch; }
                    }
            std::printf("%-10s %6d %6d %4d %9.2f %8.1f %7.2f  %s %2d,%d,%2d ch %2d %.1e\n", s.name, s.n_in, s.n_out, nt, best * 1e6,
                        mb / best / 1e9, mb / 1e6, bfam == 4 ? "ilx" : bfam == 3 ? "ilm" : bfam == 2 ? "ilt" : bfam ? "ilr" : "il ", bks, bun, bg, bch, best_err);
        }
        cx.free(W);
        cx.free(x);
        cx.free(y);
    }
    return 0;
}
