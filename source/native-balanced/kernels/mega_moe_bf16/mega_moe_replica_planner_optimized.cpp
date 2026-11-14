// Independent ABI/SO; frozen control/device planners remain byte-for-byte source.
#define DEEPEP_DEVICE_PLANNER 1
#define DEEPEP_OPTIMIZED_PLANNER 1
#define mega_moe_replica_planner_control_kernel mega_moe_replica_planner_optimized_kernel
#define mega_moe_replica_planner_control_do mega_moe_replica_planner_optimized_do
#include "mega_moe_replica_planner_control.cpp"
