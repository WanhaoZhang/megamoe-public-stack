# Replicated-expert early DW1: fixed-capture validation

The tested versions used default **0** and explicitly selected the candidate.
On 2026-10-30 the user requested compatible S9 panel builds to default to 1;
that later configuration change adds no new device or full-model evidence.
With the original rho6 placement replayed, the
candidate passed the declared canonical operator checks and paired performance
criteria. Median paired wall-time reductions were **6.025% for complete B** and
**3.837% for direct FB**. This is not full-model training acceptance.

## Code and execution scope

The tested code is based on `91fa98d77bb66c295e07686e4941bc88012a8adb`.
`MEGAFLUX_BACKWARD_REPLICATED_DW1_EARLY=1` changes active replicated experts to
DW2 -> DW1 -> DX on both owners and remote replicas. Other DW1 remains in the
tail. Expert traversal, planner, owner task order, full-K accumulation,
contributor summation order, restore and final synchronization are unchanged.
See [build, runtime guards and rollback](backward-replicated-dw1-early.md).

Validation ran on pod6-1/rack06os1 in a dedicated container, with CANN 9.2,
driver 25.1.rc3.1, Python 3.12.13, torch 2.10.0+cpu, torch_npu 2.10.0.post6 and
NumPy 1.26.4. The original capture and 13 binary runtime dependencies retained
their hashes during migration. All valid cases passed pre/post device health,
process and root-fuser gates; the final release gate passed.

Capture: world8, EP4 x expert-FSDP2, E8/Top2, M8192/H7168/F3072,
home2/slots2/capacity20480, SwiGLU limit10. Placement was replayed outside the
timed region. Measured single-layer bridge B/FB includes the corresponding
refresh, restore, compute, owner reduction and completion costs; it does not
include actual full-model FSDP or optimizer costs.

## Correctness and lifecycle

| Check | Result |
|---|---|
| Default-off kernel path | Removing only the new macro blocks reproduces the baseline header after whitespace normalization |
| CPU / build identity | 16 local CPU tests; server bridge CPU combinations; four builds, actual getters and source/SO hashes passed |
| EP2/4/8, checkpoint off/on | 12 C0/C1 process groups, 56 rank exits, 336 layer-update-rank five-output reference checks, 168 C0/C1 bitwise pairs |
| Repeated use / changing hotspots / SGD | Two-layer shared pool, balanced/hot/rotate, three SGD updates, overwrite sentinels and weight restoration passed |
| Small-shape corners | Multiple replicated logical experts, empty owner/expert/slot, token tails, zero-tile AICs; canonical no-replica path passed |
| Original capture | C0/C1 each ran three calls on eight ranks; all 48 rank-calls matched the original Y/dX/dp/dW1/dW2 hashes from the first call |
| TimerOFF/ON pairing | Three calls per variant/mode; ON outputs matched their OFF reference bitwise |

The existing full-pipeline launcher rejects legacy weights, including the
no-replica case. Legacy validation therefore checks ordinary-native reference
finiteness and the original full-pipeline rejection contract. **Candidate
legacy numerics are not qualified.** No tolerance was relaxed.

## TimerOFF performance

Each process warmed up 10 times and measured 30 samples for B and direct FB.
Each sample is the maximum wall time across eight ranks. Timer, profiler and
diagnostic readback were disabled; five-output hashes were checked before and
after the timed regions. The schedule was C0, C0, C0, C1, C1, C0, C0, C1, C1,
C0: two C0 self-repeats, then four prescribed forward/reverse pairs.

| Run | Variant | B median ms | B P90 ms | FB median ms | FB P90 ms |
|---|---|---:|---:|---:|---:|
| 00 | C0 | 25.031330 | 25.180944 | 35.686870 | 35.980395 |
| 01 | C0 | 24.949100 | 25.081487 | 35.375265 | 35.776444 |
| 02 | C0 | 24.970705 | 25.247991 | 35.003670 | 35.305908 |
| 03 | C1 | 23.455115 | 23.665918 | 33.746165 | 33.865671 |
| 04 | C1 | 23.429510 | 23.529136 | 33.507770 | 33.741958 |
| 05 | C0 | 24.964355 | 25.260298 | 34.993080 | 35.132006 |
| 06 | C0 | 24.885000 | 24.974050 | 34.885250 | 35.011504 |
| 07 | C1 | 23.396880 | 23.618663 | 33.491330 | 33.634358 |
| 08 | C1 | 23.627605 | 23.865432 | 33.808390 | 34.023237 |
| 09 | C0 | 25.005305 | 25.469918 | 35.099560 | 35.190350 |

| Pair execution order | C0/C1 run | B reduction | FB reduction |
|---|---|---:|---:|
| C0 -> C1 #1 | 02/03 | 6.069% | 3.592% |
| C1 -> C0 #1 | 05/04 | 6.148% | 4.245% |
| C0 -> C1 #2 | 06/07 | 5.980% | 3.996% |
| C1 -> C0 #2 | 09/08 | 5.510% | 3.679% |

C0 fluctuation, defined as the range of all six C0 run medians divided by their
median, was B 0.586% / FB 2.287%. All four B pairs improved and their median
gain exceeded max(2%, twice B fluctuation) = 2%. FB median gain was positive;
every FB pair and every B/FB P90 improved. The declared retention criteria
passed. The default remained off when these measurements were made.

## TimerON mechanism and remaining limit

Both variants preserve 96 logical tracks per rank with zero positive-duration
overlap on a track. Individual intervals and enclosing scope endpoints are
retained. All comparisons below use the same rank's backward origin; no
cross-rank subtraction is used to infer communication latency.

- Replica rank2 E0 DW1 first publish: 18.787 -> 9.023 ms. Across the six replica
  participants, first publish moved earlier by 9.254-9.851 ms.
- Owner rank0 first E0 DW1 tile service: 18.710 -> 13.943 ms; all local/remote
  contributor epochs are checked before this service begins.
- Owner rank0/rank4 final reduction completion: 24.035/24.018 ->
  19.496/19.977 ms. Tails after their last local DW tile shrink from
  8.862/8.494 -> 4.799/4.983 ms.
- The tradeoff is real: owner rank0 combine and AIV0 reduction entry move later
  by about 4.2 ms; later DW2 and some worker DW1 entries also move later.
- Rank2's remaining local non-replicated DW1 finishes at 23.015 ms after its
  AIV exit synchronization has finished at 21.723 ms. This exposed AIC tail,
  also observed on rank6, limits further benefit. The trace alone does not
  establish whether DW1 is compute- or memory-bound.

TimerON mechanism changes are not formal speedup measurements. Owner service
and wait scopes are not hardware busy-time estimates. No scan, double buffer
or owner-consumption-order optimization was combined with this candidate.

## Fingerprints and raw evidence index

| Artifact | SHA256 |
|---|---|
| C0 TimerOFF worker | `4e6f23303f73d8377d9b4367e3cdad8a06630ee6be574ce020f5a8022df6b42f` |
| C0 TimerON worker | `454b4c40b4011b2db6f0c8a8f286b0c324398a3b4e248d3fe1918529c8293694` |
| C1 TimerOFF worker | `6bd136137d8111f3c9f03e07df72f96f12000129a34b4d0d4ad00c0cf1707f0c` |
| C1 TimerON worker | `fa9b5dd4159434e7b7b5b4a7b839a6c041cbb43ddc4b0262bedf0b2560413032` |
| Original capture archive | `c15e452135f584c1d9ac70cca33b42953ee7b8f62a065defcd6de729df9da045` |
| Device evidence archive | `f0f97ba2cd309e94f0bb26f3ca42382d1a0b5a132b6fe114c251c39c94ed0d42` |

Raw tensors, NPYs, traces, libraries, environments and failed attempts stay in
`D:/megamoe_train/backward-dw1-early-20261029`, with the server evidence at
`/home/z00956592/megaflux/backward-dw1-validation-node6-20261029/backward-dw1-early-20261029`.
The 104,796,372-byte device archive and all 2,107 members passed SHA checks.
Local `REPORT.md`, `DEVICE-REPORT.md`, `analysis/node6/PERFORMANCE.md`, and
`analysis/backward-comparison-20261030/ANALYSIS.md` index the detailed evidence.
They are not replaced by this Git summary.

Failed launcher-directory, CRLF-shell, legacy-contract and missing-TimerOFF-
reference attempts are retained under their original numbers. They were
corrected in new attempts; no candidate numerical mismatch was observed.
Earlier occupied-node admissions did not launch device cases.
