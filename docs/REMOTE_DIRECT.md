# STRATA_REMOTE_DIRECT: the second card serves decode windows without the host

Fork `0ics-srls/Strata`, branch `v100-4090`. Measured on a Tesla V100 PCIe 32 GB (sm_70) beside an RTX 4090 (sm_89),
both PCIe 3.0 x8, no P2P, Ryzen 5 5600X, 62 GB, Qwen3.8-Flash-Next GSQ IQ3_XXS with all 24,576 experts in VRAM.

## The problem

With a second card (a `--remote-expert-opt` helper or the `--peer-device` tier) every layer of every verify window went
GPU -> host -> second GPU -> host -> GPU: the pool thread saw the doorbell, planned, staged and launched the second card's
kernels, waited for them, added the result and raised the flag the primary spins on. The primary also waited for the
host's plan (flag A) before computing its own resident experts. Profile (`STRATA_VERIFY_PROFILE`, 400 tokens): waitA 0.35,
waitB 0.23, waitCPU 1.76 ms of a 14.3 ms window.

## What it does (`STRATA_REMOTE_DIRECT=1`, opt-in)

- The second card runs a captured graph per window size. Per layer and token group: a plan kernel that first waits on
  the primary's doorbell sequence (the same mapped word the host polls), then builds the grouped-kernel plan from the
  card's own residency table; the q8_1 quantization of the published rows; `native_expert_grouped`; a reduction with the
  routing weights straight into mapped host rows, whose last block raises the card's own flag.
- The primary plans its resident entries on the device (E-6, `resident_plan` with a table of the experts the second card
  holds: those entries are no misses) and, after its combine, waits for the second card's flag and adds its rows in one
  launch.
- The CPU pool marks the second card's entries as another GPU's (zero rows) and never launches or finishes it.
- Arithmetic: the `--remote-expert-opt` path's (same activation quantization, same grouped kernel per entry, reduction in
  routing-slot order, added after the combine). Greedy outputs are bitwise identical to the helper path.
- Works over one `--expert-cache-device1` helper (with `--remote-expert-opt`) or over the `--peer-device` tier. Serial
  windows only (no layer split, no `--batch`, no `--pipeline-windows`). The second card's adaptive tier stays as loaded.

Also on this branch:

- E-6 fix: the device-planned layer 0 skipped the wait for flag 1, after which layer 1 copies the PLE rows the host
  gathers: stale rows, run-to-run different outputs. It now waits, as the all-resident graph does.
- `STRATA_SM70_MODE=<m>`: the grouped expert layout for Volta alone (`STRATA_EXP_MODE` sets every card).
- Knobs: `STRATA_REMOTE_DIRECT_GRID` (group rows of the second card's grouped launch, default 8; 0 = all, needed for
  the Volta mode 8 layout), `STRATA_REMOTE_DIRECT_PROFILE=1` (the second card's stage times per request).

## Measured (V100 + 4090, decode 400 tokens, temperature 0)

| | code | prose | explain | ms/window |
|---|---|---|---|---|
| helper, host path (`--remote-expert-opt`) | 197.1 | 115.3 | 142.5 | 14.32 |
| helper, direct, grid 8 | 200.8 | 118.5 | 145.4 | 13.56 |
| helper, direct, `STRATA_SM70_TABLE=1`, grid 0 | 204.0 | 121.2 | 147.2 | 13.40 |
| peer, direct, `STRATA_PEER_HOT=0`, `STRATA_SM70_TABLE=1`, grid 0 | 205.0 | 121.5 | 147.7 | 13.35 |

`STRATA_SM70_TABLE=1` (Volta mode 8, "opt-in until confirmed on a V100"): confirmed on a V100 PCIe, bitwise identical
outputs, faster than modes 0, 2, 4, 6, 7 here.

With the peer tier (which computes its experts in prompt chunks through mapped host memory, FP16 sums): cold prompt
28k 1,328 -> 1,719 tok/s, 226k 1,874 tok/s; tool-result reads of 125-2,000 tokens 1.75-3.5x faster than the helper's
prompt path. What remains of the primary's wait is the IQ3 grouped kernel on Volta (~54 us per layer-window for 4-5
entries).
