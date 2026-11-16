#!/usr/bin/env python3
"""Render the MegaMoe BF16 backward stage timeline as a per-lane gantt.

Reads the raw per-rank timer CSV written by profile_mega_moe_bf16.py
--capture-backward-timer, decodes the AscendTimerV2 layout exactly like
parse_backward_timer.py, and draws one lane per core type
(AIC / AIV0 / AIV1) with each AIC_*/AIV_* fixed stage as a horizontal span.

The three lanes are the three cores of every AIC group (core_id % 3:
0 = AIC, 1 = AIV0, 2 = AIV1).  A stage only gets a block on the lanes that
actually execute it; a lane with no block for that stage is idle there, so
AIC-only spin waits show up as AIC blocks with the AIV lanes empty beside
them — exactly the "AIC waiting on AIV" pattern.  The per-expert blocks
(AIC_GMM2GRAD_0..N etc.) are not drawn here; open the Perfetto trace from
trace_parser.py --phase backward for that level of detail.

Output:
  ASCII gantt printed to stdout (zero deps, no matplotlib).

Usage:
    python plot_backward_timeline.py --dir <profile_dir> --rank 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import parse_backward_timer as P

LANES = ["AIC", "AIV0", "AIV1"]

# One glyph per stage for the ASCII gantt.  Digits = AIC GEMM/dW stages
# (per-expert blocks in the Perfetto trace), letters = AIV work and waits.
PHASE_CHARS = {
    "AIV_GATHER": "a",
    "AIV_XSYNC_GATHER": "b",
    "AIV_RESTORE_TABLE": "c",
    "AIC_SPIN_AIV1": "f",
    "AIC_SPIN_DZ": "g",
    "AIC_SPIN_AIV0": "h",
    "AIC_SPIN_HP": "i",
    "AIV_XSYNC_UNPERMUTE": "j",
    "AIV_UNPERMUTE": "k",
    "AIV_XSYNC_DP": "l",
    "AIV_DP_REDUCE": "r",
    "AIV_XSYNC_TOPK": "m",
    "AIV_TOPK_UNPERMUTE": "p",
    "AIV_TABLE_RESET": "q",
    # Merged per-expert windows (see expert_spans below): each name's
    # AIC_GMM2GRAD_0..N blocks collapse into one lane window so the ASCII
    # overview keeps showing compute stages.  AIC_DO_WAIT nests inside the
    # GMM2GRAD window and is only visible in the Perfetto trace.
    "AIV_DISPATCH": "d",
    "AIC_RECOMPUTE": "e",
    "AIC_GMM2GRAD": "1",
    "AIC_GMM1GRAD": "2",
    "AIC_DW1": "3",
    "AIC_DW2": "4",
    "AIV_SWIGLU_GRAD": "s",
    "AIV_COMBINE": "t",
    "AIV_P0_TAIL": "u",
    "AIV_RECOMPUTE_PASS": "v",
    "AIV_P1_PASS": "w",
    "AIC_P2_TAIL": "x",
    "AIC_INIT": "y",
    "AIV_INIT": "z",
}

# Stages that are pure waits: AIC spins on a GM flag plus the four
# cross-rank barrier windows (AIV_XSYNC_*).  Everything else is
# compute/data movement.  Grouped under [WAIT] / [COMPUTE] in the legend
# so the gantt reads as "which wait is eating the timeline".
WAIT_PHASES = {
    "AIC_SPIN_AIV1",
    "AIC_SPIN_DZ",
    "AIC_SPIN_AIV0",
    "AIC_SPIN_HP",
    "AIC_P2_TAIL",
    "AIV_XSYNC_GATHER",
    "AIV_XSYNC_UNPERMUTE",
    "AIV_XSYNC_DP",
    "AIV_XSYNC_TOPK",
}


def lane_of_core(core_id: int) -> str:
    return LANES[core_id % 3]


def expert_spans(
    values: list[int], groups: int
) -> dict[str, dict[str, list[tuple[float, float, float]]]]:
    """lane -> expert-type -> per-core windows over that type's expert blocks.

    Decodes the shared dynamic slots exactly like parse_backward_timer.py
    and merges nothing: each core contributes one span per expert type
    (min start / max end over its AIC_GMM2GRAD_0..N blocks).  lane_blocks
    then folds the cores into a single lane window, which is what the ASCII
    overview needs — the per-block detail stays in the Perfetto trace.
    """
    dynamic = values[P.BASE_COUNTERS :]
    out = {lane: {} for lane in LANES}
    for type_index, name in enumerate(P.DYNAMIC_NAMES):
        if name not in PHASE_CHARS:
            continue  # AIC_DO_WAIT nests inside the GMM2GRAD window
        max_iter = max(
            dynamic[
                core * P.DYNAMIC_ITER_PER_CORE
                + P.BACKWARD_DYNAMIC_BASE
                + type_index
            ]
            for core in range(P.CORE_COUNT)
        )
        if max_iter <= 0:
            continue
        for core_id in range(3 * groups):
            lane = lane_of_core(core_id)
            spans = []
            for iteration in range(min(max_iter, P.MAX_DYNAMIC_ITER)):
                logical_index = (
                    P.SHARED_FIXED_COUNT
                    + (P.BACKWARD_DYNAMIC_BASE + type_index)
                    * P.MAX_DYNAMIC_ITER
                    + iteration
                )
                start, end, dur = P.event_pair(values, core_id, logical_index)
                if end > start:
                    spans.append((start, end, dur))
            if spans:
                s = min(x[0] for x in spans) / P.CYCLE_TO_US
                e = max(x[1] for x in spans) / P.CYCLE_TO_US
                d = max(x[2] for x in spans)
                out[lane].setdefault(name, []).append((s, e, d))
    return out


def collect_spans(
    values: list[int], groups: int
) -> dict[str, dict[str, list[tuple[float, float, float]]]]:
    """lane -> phase -> list of (start_us, end_us, dur_us) over participating cores.

    Only cores with a nonzero [start, end] contribute; a core that did not run
    a phase leaves that phase absent for its lane.
    """
    out = {lane: {name: [] for name in P.FIXED_NAMES} for lane in LANES}
    for core_id in range(3 * groups):
        lane = lane_of_core(core_id)
        for slot, name in enumerate(P.FIXED_NAMES):
            start, end, dur = P.event_pair(values, core_id, slot)
            if end > start:
                out[lane][name].append((start / P.CYCLE_TO_US, end / P.CYCLE_TO_US, dur))
    for lane, expert_types in expert_spans(values, groups).items():
        for name, items in expert_types.items():
            out[lane][name] = items
    return out


def lane_blocks(
    spans: dict[str, dict[str, list[tuple[float, float, float]]]]
) -> dict[str, dict[str, tuple[float, float, float]]]:
    """lane -> phase -> one merged (start, end, max_dur) block per lane.

    Each lane runs its phases in sequence, so per-lane phases do not overlap;
    merging the participating cores' spans with [min_start, max_end] gives the
    lane's overall window for that phase.
    """
    blocks = {}
    for lane, phases in spans.items():
        blocks[lane] = {}
        for name, items in phases.items():
            if not items:
                continue
            start = min(i[0] for i in items)
            end = max(i[1] for i in items)
            max_dur = max(i[2] for i in items)
            blocks[lane][name] = (start, end, max_dur)
    return blocks


def render_text(blocks, width: int = 110) -> str:
    starts = [s for lane in blocks.values() for s, _, _ in lane.values()]
    ends = [e for lane in blocks.values() for _, e, _ in lane.values()]
    if not starts:
        return "[no timing data]"
    t0, t1 = min(starts), max(ends)
    span = t1 - t0
    if span <= 0:
        return "[timing data spans zero — likely timer disabled]"

    def col(t_us: float) -> int:
        return int((t_us - t0) / span * (width - 1))

    lines = []
    header = " " * 6 + "".join(
        str((t0 + span * c / (width - 1)) // 1).rjust(1) if c % 10 == 0 else " "
        for c in range(width)
    )
    lines.append(header)
    for lane in LANES:
        row = [" "] * width
        for name, (s, e, _) in blocks[lane].items():
            if name not in PHASE_CHARS:
                continue  # KERNEL is the whole-kernel slot; skip on the gantt
            ch = PHASE_CHARS[name]
            for c in range(col(s), min(col(e) + 1, width)):
                if row[c] == " ":
                    row[c] = ch
        lines.append(f"{lane:<5}|" + "".join(row) + "|")
    lines.append("-" * (6 + width + 1))

    # Legend grouped as [COMPUTE] then [WAIT], each entry as "<glyph>=<name>".
    # Iterate PHASE_CHARS (not FIXED_NAMES) so the merged per-expert windows
    # are listed too, and show the row glyph instead of a separate number.
    legend = []
    for group, title in (("compute", "[COMPUTE]"), ("wait", "[WAIT]")):
        group_entries = []
        for name in PHASE_CHARS:
            if (name in WAIT_PHASES) != (group == "wait"):
                continue
            lane_times = []
            for lane in LANES:
                if name in blocks[lane]:
                    lane_times.append(f"{lane}={blocks[lane][name][2]:.1f}")
            if lane_times:
                group_entries.append(
                    f"{PHASE_CHARS[name]}={name}[{' '.join(lane_times)}]"
                )
        if group_entries:
            legend.append(title)
            legend.extend(group_entries)
    for i in range(0, len(legend), 2):
        lines.append("   ".join(legend[i : i + 2]))
    lines.append(f"   time window {t0:.1f}us .. {t1:.1f}us (total {span:.1f}us)")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument(
        "--seq", type=int, default=None,
        help="filter by seq when run_speed wrote seq-prefixed CSVs "
             "(seqN_rank_N_backward_time.csv); omit for the original "
             "rank_N naming from profile_mega_moe_bf16.py")
    parser.add_argument(
        "--groups",
        type=int,
        default=32,
        help="number of AIC groups to read (block_dim / GetCoreNumAic)",
    )
    parser.add_argument("--width", type=int, default=110, help="ASCII gantt width")
    args = parser.parse_args()

    # Original naming (profile_mega_moe_bf16.py) or seq-prefixed (run_speed).
    if args.seq is not None:
        raw_path = args.dir / f"seq{args.seq}_rank_{args.rank}_backward_time.csv"
    else:
        raw_path = args.dir / f"rank_{args.rank}_backward_time.csv"
    if not raw_path.exists():
        raise SystemExit(f"[SKIP] {raw_path} missing")
    values = P.load_raw(raw_path)
    spans = collect_spans(values, args.groups)
    blocks = lane_blocks(spans)
    print(render_text(blocks, args.width))


if __name__ == "__main__":
    main()
