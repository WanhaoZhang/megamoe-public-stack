// Keep the v2 policy implementation identical; only its output/control ABI differs.
#define DEEPEP_DEVICE_PLANNER 1
#define mega_moe_replica_planner_control_kernel mega_moe_replica_planner_device_kernel
#define mega_moe_replica_planner_control_do mega_moe_replica_planner_device_do
#include "mega_moe_replica_planner_control.cpp"
