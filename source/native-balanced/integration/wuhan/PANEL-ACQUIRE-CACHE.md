# Experimental DX panel acquire cache

The original rho6/rho3 experiment report, frozen device scripts, input/build
fingerprints and replay procedure are in
[the reproduction bundle](reproductions/rho63-panel-cache-20261031/README.md).

`MEGAFLUX_RESTORE_PANEL_ACQUIRE_CACHE=0/1` defaults to **0**. It requires panel restore (`MEGAFLUX_BACKWARD_RESTORE=2`) and actual B-operand readiness (`MEGAFLUX_RESTORE_OPERAND_READY=1`). Existing C1 and its build guards remain unchanged. This is independent of the rejected DW1-retire-fill scheduling experiment.

In `GMM1GradCombineGM`, each AIC owns one `PanelAcquireCache` for one expert call. The cache travels with `ReadyOperand` through CATLASS `GetTile` to the actual GM-to-L1 B copy, including prefetch. The first use still calls the original `WaitPanel`: every producer must publish the exact epoch after its payload writes retire, followed by the original DSB. Later tiles can reuse that completed acquire because this invocation does not overwrite the restored panel.

The cache key includes the ready-buffer address, slot, projection, both panel dimensions, and epoch. A key change clears the bits. Indices outside [0,64) use the original checks. The cache dies at the end of this expert's call; it never survives another layer, pool reuse, SGD update, checkpoint replay, or kernel invocation. Earlier dH operand checks and the outer first-panel check are unchanged.

The original producer lifecycle also remains: AIV1 completes panel restoration before joining P1; the all-AIV dZ/Hp retirement check precedes this GM dX stage. No new AIV UB, GM workspace, public tensor shape, barrier, arithmetic, tile traversal, or owner contributor order is introduced.

Build matched diagnostic libraries from a fresh prefix:

```bash
export MEGAFLUX_BACKWARD_REPLICATED_DW1_EARLY=1
export MEGAFLUX_RESTORE_PANEL_ACQUIRE_CACHE=1
bash scripts/build_megaflux_timer_capture.sh unique-panel-cache-prefix
```

The read-only SO getter `megaflux_restore_panel_acquire_cache()` and both build manifests record the actual selection. Use flag0 or the prior current-C1 SO for rollback. Keep TimerOFF and TimerON libraries in separate processes and verify their numerical outputs before performance conclusions.

CPU contract test (actual helper; no CANN dependency):

```bash
g++ -std=c++17 -O2 -fsanitize=undefined -fsanitize-undefined-trap-on-error \
  -Ikernels/mega_moe_bf16/include \
  test/megamoe_bf16_test/test_panel_acquire_cache.cpp -o /tmp/panel-acquire-cache-test
/tmp/panel-acquire-cache-test
```

Validation scope is the two original rho6/rho3 captures, world8/two EP4 groups, E8/Top2, M8192/H7168/F3072, home2/slots2, capacity20480, BF16 saved and limit10. Required evidence includes current-C1 bitwise outputs, OFF/ON equality, pool poisoning/restoration, changing hot experts, SGD/checkpoint reuse, unchanged tile-coordinate coverage, serial 96-lane Timer exports, and same-node paired TimerOFF complete B/direct FB. This does not establish other shapes/topologies or full-network acceptance.
