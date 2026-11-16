# Single-microbatch F/B/owner planner research

Source baseline: `D:/megamoe_train/planner-executor-ep8-lab`.
Working project: `D:/megamoe_train/single-mb-fbw-planner-study`.
All three source repositories use `research/single-mb-fbw-planner`.

## Reproduce local diagnostics

Use a project-local Python environment with NumPy, Matplotlib and Ziglang.
Run from the project root; use `python -I -m ziglang` to prevent the unrelated
parent directory's `ziglang` package from shadowing the compiler.

```
python tools/bootstrap.py --baseline D:/megamoe_train/planner-executor-ep8-lab
python tools/analyze_existing.py --baseline D:/megamoe_train/planner-executor-ep8-lab
python tools/budget_scan.py --baseline D:/megamoe_train/planner-executor-ep8-lab
python -I -m ziglang c++ -shared -O2 -I source/native-balanced/include tools/policy_oracle.cpp -o analysis/policy_oracle.dll
python tools/test_budget.py D:/megamoe_train/single-mb-fbw-planner-study/analysis/policy_oracle.dll
python -I -m ziglang c++ -shared -O2 -I source/native-balanced/include tools/step_time_oracle.cpp -o analysis/step_time_oracle.dll
python tools/offline_search.py --baseline D:/megamoe_train/planner-executor-ep8-lab --library D:/megamoe_train/single-mb-fbw-planner-study/analysis/step_time_oracle.dll
python tools/plot_existing.py --baseline D:/megamoe_train/planner-executor-ep8-lab
```

Bootstrap verifies the full frozen manifest before cloning and refuses to
overwrite existing source checkouts. The offline scorer is inherited, imperfect,
and not requalified for the new candidate bank. Its scores are hypotheses only.

`eplb_budget_N` requires rebuilding both the native Host extension and planner SO.
It is a budget-intervention baseline, not a qualified new owner-aware planner.
The independent pod5-8 runtime has passed the common plan gate and ten native
qualification jobs, including three-update full-model runs. These are numerical
qualification, not steady-state performance or natural-route quality results.
The matched / wide pipeline workers passed18 gates and completed20 full-update
screen cells without a2% gain. Six paired prepared-plan gates,22 causal timing
cells and12 owner-window gates plus4 online screen cells also completed.
See source/native-balanced/docs/single-mb-causal-results.md for the scoped
findings. Four redispatch dHp runahead variants are building; their29 gates and
strong-baseline endpoint screens run only after all CPU compilers retire.
The new online owner-aware planner and strong repeated comparisons remain
pending. See STATUS.json and REPORT.md for exact evidence and limitations.

The portable kit is mirrored under `research/single_mb_fbw` in the native Git
branch. To use that kit, copy its tools and documents into a fresh project root;
keep the three source repositories under `source/` as shown above. Do not run
inherited controllers with their hardcoded old project or node paths.
