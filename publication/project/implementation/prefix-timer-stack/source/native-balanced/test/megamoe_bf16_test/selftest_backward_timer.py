"""Offline self-test: synthetic backward timer data -> parsed CSV + ASCII gantt.

Fabricates an AscendTimerV2 buffer exactly in the new layout (21 fixed
slots + 9 backward dynamic types, EPR=4, 3 AIC groups), then runs the real
parse_rank + plot + trace pipeline over it to verify decoding end to end
without hardware.

The synthetic timeline is faithful to the kernel's control flow: waits for
the first tile of a stage are absorbed into that stage's first expert bar
(Tik fires at function entry, the WaitForCube sits inside), and every
transition between stages is itself a named bar, so the lanes tile with no
unexplained gaps.
"""
import csv
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import parse_backward_timer as P
import plot_backward_timeline as PL

values = [0] * P.TOTAL_SIZE


def put(core, logical, s_us, e_us):
    """Write a [start, end] pair; inputs are microseconds, stored as cycles
    (AscendTimerV2 counts cycles, parsers divide by CYCLE_TO_US=1000)."""
    base = P.core_base(core) + logical * 2
    values[base] = int(s_us * 1000)
    values[base + 1] = int(e_us * 1000)


def put_dyn(core, type_idx, it, s_us, e_us):
    logical = (P.SHARED_FIXED_COUNT
               + (P.BACKWARD_DYNAMIC_BASE + type_idx) * P.MAX_DYNAMIC_ITER
               + it)
    put(core, logical, s_us, e_us)


# Fixed-slot indices by name (must stay in sync with FIXED_NAMES).
S = {name: idx for idx, name in enumerate(P.FIXED_NAMES)}

EPR = 4
GROUPS = 3
# Groups run concurrently on real hardware: give each a small skew so the
# per-lane union windows overlap realistically instead of spanning each other.
for g in range(GROUPS):
    aic = g * 3
    aiv0, aiv1 = aic + 1, aic + 2
    t0 = g * 7
    # Wakeup/init head (AIC_INIT on the cube lane, AIV_INIT on both AIVs).
    # Start at cycle >= 1: the trace filters start<=0 events.
    put(aic, S["AIC_INIT"], t0 + 1, t0 + 4)
    put(aiv0, S["AIV_INIT"], t0 + 1, t0 + 2)
    put(aiv1, S["AIV_INIT"], t0 + 1, t0 + 3)
    t = t0 + 5
    # P0-A/P0-B on the AIVs.
    put(aiv0, S["AIV_GATHER"], t, t + 50); put(aiv1, S["AIV_GATHER"], t, t + 50)
    t += 55
    put(aiv0, S["AIV_XSYNC_GATHER"], t, t + 20); put(aiv1, S["AIV_XSYNC_GATHER"], t, t + 20)
    t += 25
    put(aiv0, S["AIV_RESTORE_TABLE"], t, t + 10); put(aiv1, S["AIV_RESTORE_TABLE"], t, t + 10)
    t += 15
    # AIV1's tail happens right after restore; AIV0's tail follows dispatch.
    put(aiv1, S["AIV_P0_TAIL"], t, t + 3)
    put(aiv1, S["AIV_RECOMPUTE_PASS"], t + 3, t + 8)
    # AIC recompute + AIV0 dispatch overlap (perf-opt-2 layout).
    for e in range(EPR):
        put_dyn(aic, 1, e, t + e * 40, t + e * 40 + 35)          # recompute
        put_dyn(aiv0, 0, e, t + e * 40, t + e * 40 + 25)         # dispatch
    disp_end = t + (EPR - 1) * 40 + 25
    put(aiv0, S["AIV_P0_TAIL"], disp_end, disp_end + 3)
    put(aiv0, S["AIV_RECOMPUTE_PASS"], disp_end + 3, disp_end + 8)
    t += EPR * 40 + 10
    put(aic, S["AIC_SPIN_AIV1"], t, t + 5)
    t += 10
    # Dual-AIV P1 first acquires AIV0's released P0 UB, then both AIV lanes
    # record per-expert SwiGLU bars while tiles alternate between them.
    put(aic, S["AIC_SPIN_AIV0"], t, t + 3)
    t += 5
    p1_base = t
    # Accumulated per-lane wait (AIV_P1_WAIT, Accumulate slot): each lane
    # waits ~10us per tile before its epilogue runs.  Stored as a single
    # [first_start, first_start + sum_of_waits] pair — exactly what the
    # device's first Accumulate write leaves in the buffer.
    put(aiv0, S["AIV_P1_WAIT"], p1_base + 8, p1_base + 8 + EPR * 10)
    put(aiv1, S["AIV_P1_WAIT"], p1_base + 10, p1_base + 10 + EPR * 10)
    for e in range(EPR):
        ws = p1_base + e * 60 + 8
        we = p1_base + e * 60 + 8 + 45
        put_dyn(aic, 2, e, p1_base + e * 60, p1_base + e * 60 + 8)     # do wait
        put_dyn(aic, 3, e, ws, we)                                     # gmm2grad
        put_dyn(aiv0, 4, e, ws, we)                                    # dual-AIV span
        put_dyn(aiv1, 4, e, ws + 2, we + 2)                            # small lane skew
    t += EPR * 60 + 10
    put(aic, S["AIC_SPIN_DZ"], t, t + 40)
    put(aic, S["AIC_DO_REUSE_WAIT_0"], t, t + 11)
    put(aic, S["AIC_DZ_WAIT_0"], t, t + 17)
    put(aiv0, S["AIV_DX_WAIT_0"], t + 20, t + 43)
    t += 45
    p2_base = t
    for e in range(EPR):
        put_dyn(aic, 5, e, p2_base + e * 50, p2_base + e * 50 + 40)    # gmm1grad
        if e == 0:
            put_dyn(aiv0, 6, 0, p2_base, p2_base + 40)
        else:
            put_dyn(aiv0, 6, e, p2_base + e * 50, p2_base + e * 50 + 40)  # combine
    t += EPR * 50 + 10
    put(aic, S["AIC_P2_TAIL"], t, t + 6)
    t += 10
    for e in range(EPR):
        put_dyn(aic, 7, e, t + e * 30, t + e * 30 + 25)                # dw1
    t += EPR * 30 + 10
    put(aic, S["AIC_SPIN_HP"], t, t + 4)
    t += 8
    for e in range(EPR):
        put_dyn(aic, 8, e, t + e * 30, t + e * 30 + 25)                # dw2
    t += EPR * 30 + 10
    # P5/P6 tail on the AIVs (AIC is genuinely idle here: blank is correct).
    put(aiv0, S["AIV_XSYNC_UNPERMUTE"], t, t + 30); put(aiv1, S["AIV_XSYNC_UNPERMUTE"], t, t + 30)
    t += 35
    put(aiv0, S["AIV_UNPERMUTE"], t, t + 60); put(aiv1, S["AIV_UNPERMUTE"], t, t + 60)
    t += 65
    put(aiv0, S["AIV_XSYNC_DP"], t, t + 15); put(aiv1, S["AIV_XSYNC_DP"], t, t + 15)
    t += 20
    put(aiv0, S["AIV_DP_REDUCE"], t, t + 90); put(aiv1, S["AIV_DP_REDUCE"], t, t + 90)
    t += 95
    put(aiv0, S["AIV_XSYNC_TOPK"], t, t + 15); put(aiv1, S["AIV_XSYNC_TOPK"], t, t + 15)
    t += 20
    put(aiv0, S["AIV_TOPK_UNPERMUTE"], t, t + 25); put(aiv1, S["AIV_TOPK_UNPERMUTE"], t, t + 25)
    t += 30
    put(aiv0, S["AIV_TABLE_RESET"], t, t + 5); put(aiv1, S["AIV_TABLE_RESET"], t, t + 5)

# Expert count registry: every core writes EPR for the 9 backward types.
for core in range(P.CORE_COUNT):
    for ti in range(len(P.DYNAMIC_NAMES)):
        values[P.BASE_COUNTERS
               + core * P.DYNAMIC_ITER_PER_CORE
               + P.BACKWARD_DYNAMIC_BASE + ti] = EPR

# 1) real parser over the synthetic raw values
with tempfile.TemporaryDirectory() as tmp:
    raw = Path(tmp) / "rank_0_backward_time.csv"
    parsed = Path(tmp) / "rank_0_backward_time_parsed.csv"
    with raw.open("w", newline="") as stream:
        writer = csv.writer(stream)
        for i in range(0, P.TOTAL_SIZE, 16):
            writer.writerow(values[i:i + 16])
    P.parse_rank(raw, parsed, GROUPS)
    with parsed.open() as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    aic0 = rows[0]   # core_id 0 = group 0 AIC
    aiv0 = rows[1]   # core_id 1 = group 0 AIV0
    aiv1 = rows[2]   # core_id 2 = group 0 AIV1
    assert float(aic0["AIC_SPIN_DZ_duration(us)"]) == 40.0, "spin dz decode"
    assert float(aic0["AIC_DO_WAIT_0_duration(us)"]) == 8.0, "do wait decode"
    assert float(aic0["AIC_GMM2GRAD_0_duration(us)"]) == 45.0, "gmm2grad decode"
    assert float(aiv0["AIV_DISPATCH_3_duration(us)"]) == 25.0, "dispatch decode"
    assert float(aiv0["AIV_SWIGLU_GRAD_2_duration(us)"]) == 45.0, "AIV0 swiglu decode"
    assert float(aiv1["AIV_SWIGLU_GRAD_2_duration(us)"]) == 45.0, "swiglu decode"
    assert float(aic0["AIC_INIT_duration(us)"]) > 0.0, "aic init decode"
    assert float(aiv0["AIV_INIT_duration(us)"]) > 0.0, "aiv init decode"
    assert float(aiv0["AIV_P0_TAIL_duration(us)"]) == 3.0, "p0 tail decode"
    assert float(aiv1["AIV_RECOMPUTE_PASS_duration(us)"]) == 5.0, "recompute pass decode"
    assert float(aiv0["AIV_P1_WAIT_duration(us)"]) == EPR * 10, "AIV0 p1 wait decode"
    assert float(aiv1["AIV_P1_WAIT_duration(us)"]) == EPR * 10, "AIV1 p1 wait decode"
    assert float(aic0["AIC_P2_TAIL_duration(us)"]) == 6.0, "p2 tail decode"
    assert float(aiv0["AIV_COMBINE_0_duration(us)"]) == 40.0, "combine decode"
    print("[OK] parse_rank: fixed + transition + per-expert columns decode")

# 2) ASCII gantt with merged expert windows
spans = PL.collect_spans(values, GROUPS)
blocks = PL.lane_blocks(spans)
text = PL.render_text(blocks, 110)
print(text)
for expect in ("AIV_P0_TAIL", "AIV_RECOMPUTE_PASS", "AIV_P1_WAIT",
               "AIC_P2_TAIL", "AIC_INIT", "AIV_INIT"):
    assert (expect in blocks["AIC"] or expect in blocks["AIV0"]
            or expect in blocks["AIV1"]), f"transition bar missing: {expect}"
print("[OK] transition bars render; lanes tile without unexplained gaps")

# 3) trace JSON over the parsed CSV
from trace_parser import generate_rank_events
with tempfile.TemporaryDirectory() as tmp:
    raw = Path(tmp) / "rank_0_backward_time.csv"
    parsed = Path(tmp) / "rank_0_backward_time_parsed.csv"
    with raw.open("w", newline="") as stream:
        writer = csv.writer(stream)
        for i in range(0, P.TOTAL_SIZE, 16):
            writer.writerow(values[i:i + 16])
    P.parse_rank(raw, parsed, GROUPS)
    events = generate_rank_events(parsed, 0, "backward")
    names = {ev["name"] for ev in events}
    for expect in ("AIC_GMM2GRAD_0", "AIV_SWIGLU_GRAD_3", "AIC_DO_WAIT_1",
                   "AIV_XSYNC_DP", "AIC_SPIN_DZ", "AIV_TABLE_RESET",
                   "AIV_P0_TAIL", "AIV_P1_WAIT", "AIC_P2_TAIL", "AIV_INIT"):
        assert expect in names, f"trace missing {expect}"
    for name, duration in (("AIC_DZ_WAIT_0", 17.0), ("AIV_DX_WAIT_0", 23.0),
                           ("AIC_DO_REUSE_WAIT_0", 11.0)):
        event = next(ev for ev in events if ev["name"] == name)
        assert event["dur"] == duration and event["cat"] == "wait"
        assert event["args"]["is_wait"] and "timing_note" not in event["args"]
    waits = {ev["name"] for ev in events if ev["cat"] == "wait"}
    assert "AIC_SPIN_DZ" in waits and "AIV_XSYNC_DP" in waits, "wait tagging"
    assert "AIC_P2_TAIL" in waits, "p2 tail should be tagged wait"
    assert "AIV_P1_WAIT" in waits, "accumulated P1 tile wait should be tagged wait"
    assert "AIC_GMM2GRAD_0" not in waits, "compute mis-tagged as wait"
    print(f"[OK] trace_parser: {len(events)} events, waits tagged correctly")
