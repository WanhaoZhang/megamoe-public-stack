# Physical expert capacity

The default build supports 32 local physical experts. MoonEP planning uses equal home and replica counts. EP8/E256 needs 32 homes plus 32 replica slots and a 64-capacity build. This is an implementation capacity, not a MoonEP algorithm limit.

Set ASCEND_DEEPEP_MAX_LOCAL_EXPERTS=64 for setup.py and scripts/build_megaflux_worker_pipeline.sh, and MEGAMOE_MAX_LOCAL_EXPERTS=64 in the core CMake configuration. Rebuild Host, planning libraries and the worker using the same capacity, SDK/compiler and restore feature flags. Runtime checks reject different Host/planner/worker capacities. Keep 32-capacity products in separate directories.

The pipeline build stages ascend_deepep_megaflux_worker_bf16 under the public library filename libascend_deepep_megaflux_early_bf16.so. The generic early_bf16 CMake target is a different operator.

The private shared-backward device kernel uses a uint64 split scalar so the following 8-byte-aligned descriptor is packed correctly by the new CANN frontend. Exported Host wrappers retain their uint32 ABI.
