cmake_minimum_required(VERSION 3.16)

# CANN environment.
if(NOT DEFINED ASCEND_HOME_PATH)
    if(DEFINED ENV{ASCEND_HOME_PATH})
        set(ASCEND_HOME_PATH "$ENV{ASCEND_HOME_PATH}")
    else()
        message(FATAL_ERROR "ASCEND_HOME_PATH is not set. Source the CANN set_env.sh first.")
    endif()
endif()

set(CCEC "${ASCEND_HOME_PATH}/tools/bisheng_compiler/bin/bisheng" CACHE FILEPATH "Bisheng compiler")
if(NOT EXISTS "${CCEC}")
    message(FATAL_ERROR "Bisheng compiler not found: ${CCEC}")
endif()

if(NOT DEFINED CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER "${CCEC}" CACHE FILEPATH "CXX compiler")
endif()

project(ascend_deepep LANGUAGES CXX)
set(MEGAMOE_MAX_LOCAL_EXPERTS 32 CACHE STRING "Physical local expert capacity: 32 or 64")
if(NOT MEGAMOE_MAX_LOCAL_EXPERTS MATCHES "^(32|64)$")
    message(FATAL_ERROR "MEGAMOE_MAX_LOCAL_EXPERTS must be 32 or 64")
endif()
add_compile_definitions(MEGAMOE_MAX_LOCAL_EXPERTS=${MEGAMOE_MAX_LOCAL_EXPERTS})

get_filename_component(ASCEND_DEEPEP_CXX_COMPILER_NAME "${CMAKE_CXX_COMPILER}" NAME)
if(NOT ASCEND_DEEPEP_CXX_COMPILER_NAME MATCHES "bisheng")
    message(FATAL_ERROR
        "ascend_deepep kernels must be configured with Bisheng, but CMAKE_CXX_COMPILER="
        "${CMAKE_CXX_COMPILER}. Remove the existing build directory or pass "
        "-DCMAKE_CXX_COMPILER=${CCEC}.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(ASCEND_DEEPEP_BISHENG_ARCH "a2" CACHE STRING "Target architecture: a2 or a5")
set_property(CACHE ASCEND_DEEPEP_BISHENG_ARCH PROPERTY STRINGS a2 a5)
option(ASCEND_DEEPEP_ENABLE_TIMER "Enable kernel profiling instrumentation" OFF)
set(ASCEND_DEEPEP_BF16_SWIGLU_DECOUPLE "auto" CACHE STRING
    "Ordinary BF16 forward SwiGLU decoupling: off, on, or auto (H<3584; H=3584 only mxfp8)")
set_property(CACHE ASCEND_DEEPEP_BF16_SWIGLU_DECOUPLE PROPERTY STRINGS off on auto)
option(ASCEND_DEEPEP_BF16_GM_SWIGLU_TILE_READY
    "S9 BF16 GM SwiGLU producer tile progress (experimental)" OFF)

set(ASCEND_DEEPEP_BF16_GMM1_SCHEDULE "adaptive-auto" CACHE STRING
    "BF16 GMM1 schedule: adaptive-auto, adaptive-ready-first, adaptive-expert or legacy")
set_property(CACHE ASCEND_DEEPEP_BF16_GMM1_SCHEDULE PROPERTY STRINGS
    adaptive-auto adaptive-ready-first adaptive-expert legacy)
option(ASCEND_DEEPEP_MEGA_MOE_BF16_DEBUG_PRINT
       "Enable MegaMoe BF16 backward phase PRINTF diagnostics" OFF)
option(ASCEND_DEEPEP_BUILD_MEGA_MOE "Build standalone MegaMoe targets" OFF)
option(ASCEND_DEEPEP_BUILD_MEGA_MOE_BF16 "Build standalone MegaMoe BF16 target" OFF)
option(ASCEND_DEEPEP_BUILD_MEGA_MOE_BF16_MGROUP "Build standalone MegaMoe BF16 MGroup target" OFF)
option(ASCEND_DEEPEP_BUILD_MEGA_MOE_BF16_UDMA "Build standalone MegaMoe BF16 UDMA target" OFF)
option(ASCEND_DEEPEP_BUILD_TRANSPORT_BENCHMARK
    "Build DeepEP UDMA/SDMA/MTE transport benchmark" OFF)
option(ASCEND_DEEPEP_BUILD_MEGA_MOE_W4A8 "Build standalone MegaMoe W4A8 targets" OFF)
if(ASCEND_DEEPEP_BISHENG_ARCH STREQUAL "a5")
    set(ASCEND_DEEPEP_CATLASS_ARCH 3510)
elseif(ASCEND_DEEPEP_BISHENG_ARCH STREQUAL "a2")
    set(ASCEND_DEEPEP_CATLASS_ARCH 2201)
else()
    message(FATAL_ERROR "Unsupported ASCEND_DEEPEP_BISHENG_ARCH=${ASCEND_DEEPEP_BISHENG_ARCH}")
endif()

# SHMEM is managed as a submodule and built into 3rdparty/shmem/install by default.
set(ASCEND_DEEPEP_SHMEM_SOURCE_DIR "${CMAKE_SOURCE_DIR}/3rdparty/shmem" CACHE PATH "SHMEM source directory")
if(NOT DEFINED SHMEM_HOME_PATH)
    if(DEFINED ENV{SHMEM_HOME_PATH})
        set(SHMEM_HOME_PATH "$ENV{SHMEM_HOME_PATH}")
    else()
        set(SHMEM_HOME_PATH "${ASCEND_DEEPEP_SHMEM_SOURCE_DIR}/install")
    endif()
endif()

if(NOT EXISTS "${SHMEM_HOME_PATH}/shmem/include/shmem.h")
    message(FATAL_ERROR
        "SHMEM installation not found under ${SHMEM_HOME_PATH}. "
        "Run scripts/build_shmem.sh first.")
endif()

file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

include("${CMAKE_SOURCE_DIR}/cmake/AscendDeepEPKernel.cmake")

if(ASCEND_DEEPEP_BUILD_MEGA_MOE_BF16_UDMA)
    if(NOT ASCEND_DEEPEP_BISHENG_ARCH STREQUAL "a5")
        message(FATAL_ERROR "Standalone MegaMoe BF16 UDMA requires ASCEND_DEEPEP_BISHENG_ARCH=a5")
    endif()
    ascend_deepep_add_kernel_library(
        ascend_deepep_mega_moe_bf16_udma
        TIMER_WITHOUT_BLOCK_LOCAL_INIT
        SOURCES
            kernels/mega_moe_bf16_udma/mega_moe_bf16_udma.cpp
        INCLUDES
            include
            kernels
            kernels/mega_moe_bf16_udma/include
            kernels/mega_moe_bf16/include
            kernels/mega_moe/include
            "${SHMEM_HOME_PATH}/shmem/include"
            "${SHMEM_HOME_PATH}/shmem/src"
        COMPILE_DEFINITIONS
            CATLASS_ARCH=${ASCEND_DEEPEP_CATLASS_ARCH}
    )
    set_source_files_properties(
        kernels/mega_moe_bf16_udma/mega_moe_bf16_udma.cpp
        PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(ASCEND_DEEPEP_BUILD_TRANSPORT_BENCHMARK)
    if(NOT ASCEND_DEEPEP_BISHENG_ARCH STREQUAL "a5")
        message(FATAL_ERROR "DeepEP transport benchmark requires ASCEND_DEEPEP_BISHENG_ARCH=a5")
    endif()
    ascend_deepep_add_kernel_library(
        deepep_transport_benchmark_kernel
        SOURCES
            test/transport_benchmark/transport_benchmark_kernel.cpp
        INCLUDES
            test/transport_benchmark
            "${SHMEM_HOME_PATH}/shmem/include"
            "${SHMEM_HOME_PATH}/shmem/src"
    )
    set_source_files_properties(
        test/transport_benchmark/transport_benchmark_kernel.cpp
        PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)

    add_executable(
        deepep_transport_benchmark
        test/transport_benchmark/transport_benchmark_stub.cpp)
    target_include_directories(deepep_transport_benchmark
        PRIVATE
            test/transport_benchmark
            "${SHMEM_HOME_PATH}/shmem/include"
            "${SHMEM_HOME_PATH}/shmem/src"
            "${SHMEM_HOME_PATH}/memfabric_hybrid/include/smem/host"
            "${CMAKE_SOURCE_DIR}/3rdparty/shmem/examples/utils"
            ${ASCEND_DEEPEP_HOST_INCLUDE_DIRS})
    target_link_directories(deepep_transport_benchmark
        PRIVATE
            ${ASCEND_DEEPEP_HOST_LINK_DIRS}
            "${SHMEM_HOME_PATH}/memfabric_hybrid/lib"
            "${SHMEM_HOME_PATH}/shmem/lib")
    target_link_options(deepep_transport_benchmark
        PRIVATE
            "SHELL:-Wl,-rpath,\$ORIGIN/../lib"
            "-Wl,-rpath,${ASCEND_HOME_PATH}/lib64"
            "-Wl,-rpath,${ASCEND_HOME_PATH}/aarch64-linux/lib64"
            "-Wl,-rpath,${SHMEM_HOME_PATH}/memfabric_hybrid/lib"
            "-Wl,-rpath,${SHMEM_HOME_PATH}/shmem/lib")
    target_link_libraries(deepep_transport_benchmark
        PRIVATE
            deepep_transport_benchmark_kernel
            -Wl,--no-as-needed
            runtime
            stdc++
            ascendcl
            m
            tiling_api
            platform
            c_sec
            dl
            nnopbase
            pthread
            shmem
            -Wl,--as-needed)
endif()

# Common operators share this library. MegaMoe is intentionally excluded and
# built only by the independent ascend_deepep_mega_moe target below.
ascend_deepep_add_kernel_library(
    ascend_deepep_kernels
    SOURCES
        kernels/dispatch.cpp
        kernels/combine.cpp
        kernels/allgather_matmul.cpp
    INCLUDES
        include
        kernels
        3rdparty/catccos/include
        3rdparty/catccos/utils
        3rdparty/catccos/tools
        3rdparty/catccos/examples/ascend950_allgather_matmul
    COMPILE_DEFINITIONS
        CATLASS_ARCH=${ASCEND_DEEPEP_CATLASS_ARCH}
)

if(ASCEND_DEEPEP_BUILD_MEGA_MOE_BF16)
    if(NOT ASCEND_DEEPEP_BISHENG_ARCH STREQUAL "a5")
        message(FATAL_ERROR "Standalone MegaMoe BF16 requires ASCEND_DEEPEP_BISHENG_ARCH=a5")
    endif()
    if(ASCEND_DEEPEP_BF16_SWIGLU_DECOUPLE STREQUAL "off")
        set(BF16_SWIGLU_DECOUPLE_MODE 0)
    elseif(ASCEND_DEEPEP_BF16_SWIGLU_DECOUPLE STREQUAL "on")
        set(BF16_SWIGLU_DECOUPLE_MODE 1)
    elseif(ASCEND_DEEPEP_BF16_SWIGLU_DECOUPLE STREQUAL "auto")
        set(BF16_SWIGLU_DECOUPLE_MODE 2)
    else()
        message(FATAL_ERROR "ASCEND_DEEPEP_BF16_SWIGLU_DECOUPLE must be off, on or auto")
    endif()
    message(STATUS "MegaMoe BF16 SwiGLU decoupling: ${ASCEND_DEEPEP_BF16_SWIGLU_DECOUPLE}")
    set(BF16_GMM1_SCHEDULE_DEFINITIONS)
    if(ASCEND_DEEPEP_BF16_GMM1_SCHEDULE STREQUAL "adaptive-auto")
        list(APPEND BF16_GMM1_SCHEDULE_DEFINITIONS
            DEEPEP_BF16_GMM1_ADAPTIVE_NSTRIPE=1
            DEEPEP_BF16_GMM1_READY_FIRST=1
            DEEPEP_BF16_GMM1_TILE_READY=1
            DEEPEP_BF16_GMM1_AUTO_SELECT=1)
    elseif(ASCEND_DEEPEP_BF16_GMM1_SCHEDULE STREQUAL "adaptive-ready-first")
        list(APPEND BF16_GMM1_SCHEDULE_DEFINITIONS
            DEEPEP_BF16_GMM1_ADAPTIVE_NSTRIPE=1
            DEEPEP_BF16_GMM1_READY_FIRST=1
            DEEPEP_BF16_GMM1_TILE_READY=1)
    elseif(ASCEND_DEEPEP_BF16_GMM1_SCHEDULE STREQUAL "adaptive-expert")
        list(APPEND BF16_GMM1_SCHEDULE_DEFINITIONS
            DEEPEP_BF16_GMM1_ADAPTIVE_NSTRIPE=1)
    elseif(NOT ASCEND_DEEPEP_BF16_GMM1_SCHEDULE STREQUAL "legacy")
        message(FATAL_ERROR
            "ASCEND_DEEPEP_BF16_GMM1_SCHEDULE must be adaptive-auto, adaptive-ready-first, adaptive-expert or legacy")
    endif()
    message(STATUS "MegaMoe BF16 GMM1 schedule: ${ASCEND_DEEPEP_BF16_GMM1_SCHEDULE}")
    ascend_deepep_add_kernel_library(
        ascend_deepep_mega_moe_bf16
        TIMER_WITHOUT_BLOCK_LOCAL_INIT
        SOURCES
            kernels/mega_moe_bf16/mega_moe_bf16.cpp
            kernels/mega_moe_bf16/mega_moe_bf16_backward.cpp
        INCLUDES
            include
            kernels
            kernels/mega_moe_bf16/include
            kernels/mega_moe/include
        COMPILE_DEFINITIONS
            CATLASS_ARCH=${ASCEND_DEEPEP_CATLASS_ARCH}
            ${BF16_GMM1_SCHEDULE_DEFINITIONS}
            DEEPEP_BF16_SWIGLU_DECOUPLE=${BF16_SWIGLU_DECOUPLE_MODE}
    )
    if(ASCEND_DEEPEP_MEGA_MOE_BF16_DEBUG_PRINT)
        target_compile_definitions(
            ascend_deepep_mega_moe_bf16
            PRIVATE MEGA_MOE_BF16_DEBUG_PRINT=1)
    endif()
    set_source_files_properties(
        kernels/mega_moe_bf16/mega_moe_bf16.cpp
        PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
    set_source_files_properties(
        kernels/mega_moe_bf16/mega_moe_bf16_backward.cpp
        PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)

    add_executable(
        mega_moe_bf16_stub
        kernels/mega_moe_bf16/mega_moe_bf16_stub.cpp)
    target_include_directories(mega_moe_bf16_stub
        PRIVATE
            "${CMAKE_SOURCE_DIR}"
            "${CMAKE_SOURCE_DIR}/include"
            "${CMAKE_SOURCE_DIR}/kernels"
            "${CMAKE_SOURCE_DIR}/kernels/mega_moe_bf16/include"
            "${CMAKE_SOURCE_DIR}/kernels/mega_moe/include"
            "${CMAKE_SOURCE_DIR}/3rdparty/catlass/include"
            "${SHMEM_HOME_PATH}/shmem/include"
            "${SHMEM_HOME_PATH}/shmem/src"
            "${SHMEM_HOME_PATH}/memfabric_hybrid/include/smem/host"
            ${ASCEND_DEEPEP_HOST_INCLUDE_DIRS})
    target_link_directories(mega_moe_bf16_stub
        PRIVATE
            ${ASCEND_DEEPEP_HOST_LINK_DIRS}
            "${SHMEM_HOME_PATH}/memfabric_hybrid/lib"
            "${SHMEM_HOME_PATH}/shmem/lib")
    target_link_options(mega_moe_bf16_stub
        PRIVATE
            "SHELL:-Wl,-rpath,\\$ORIGIN/../lib"
            "-Wl,-rpath,${ASCEND_HOME_PATH}/lib64"
            "-Wl,-rpath,${ASCEND_HOME_PATH}/aarch64-linux/lib64"
            "-Wl,-rpath,${SHMEM_HOME_PATH}/memfabric_hybrid/lib"
            "-Wl,-rpath,${SHMEM_HOME_PATH}/shmem/lib")
    target_link_libraries(mega_moe_bf16_stub
        PRIVATE
            ascend_deepep_mega_moe_bf16
            -Wl,--no-as-needed
            runtime
            stdc++
            ascendcl
            m
            tiling_api
            platform
            c_sec
            dl
            nnopbase
            pthread
            shmem
            -Wl,--as-needed)
endif()
if(ASCEND_DEEPEP_BUILD_MEGA_MOE_BF16_MGROUP)
    if(NOT ASCEND_DEEPEP_BISHENG_ARCH STREQUAL "a5")
        message(FATAL_ERROR "Standalone MegaMoe BF16 MGroup requires ASCEND_DEEPEP_BISHENG_ARCH=a5")
    endif()
    ascend_deepep_add_kernel_library(
        ascend_deepep_mega_moe_bf16_mgroup
        TIMER_WITHOUT_BLOCK_LOCAL_INIT
        SOURCES
            kernels/mega_moe_bf16_mgroup/mega_moe_bf16_mgroup.cpp
        INCLUDES
            include
            kernels
            kernels/mega_moe_bf16_mgroup/include
            kernels/mega_moe_bf16/include
            kernels/mega_moe/include
        COMPILE_DEFINITIONS
            CATLASS_ARCH=${ASCEND_DEEPEP_CATLASS_ARCH}
            DEEPEP_MEGA_MOE_BF16_MGROUP_DISTINCT_MMAD_EVENTS=1
    )
    # Only MGroup uses fixed-position local tensors.  Keep this include ahead
    # of the CATLASS submodule so other kernels retain the upstream TPipe-based
    # Catlass::Arch::Resource implementation.
    target_include_directories(
        ascend_deepep_mega_moe_bf16_mgroup
        BEFORE PRIVATE
            "${CMAKE_SOURCE_DIR}/kernels/mega_moe_bf16_mgroup/catlass_overlay")
    set_source_files_properties(
        kernels/mega_moe_bf16_mgroup/mega_moe_bf16_mgroup.cpp
        PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()
if(ASCEND_DEEPEP_BUILD_MEGA_MOE_BF16_UDMA)
    message(STATUS "MegaMoe BF16 UDMA kernel library: ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libascend_deepep_mega_moe_bf16_udma.so")

    add_executable(
        mega_moe_bf16_udma_stub
        kernels/mega_moe_bf16/mega_moe_bf16_stub.cpp)
    target_compile_definitions(mega_moe_bf16_udma_stub
        PRIVATE
            DEEPEP_MEGA_MOE_STUB_BF16_UDMA=1)
    target_include_directories(mega_moe_bf16_udma_stub
        PRIVATE
            "${CMAKE_SOURCE_DIR}"
            "${CMAKE_SOURCE_DIR}/include"
            "${CMAKE_SOURCE_DIR}/kernels"
            "${CMAKE_SOURCE_DIR}/kernels/mega_moe_bf16_udma/include"
            "${CMAKE_SOURCE_DIR}/kernels/mega_moe_bf16/include"
            "${CMAKE_SOURCE_DIR}/kernels/mega_moe/include"
            "${CMAKE_SOURCE_DIR}/3rdparty/catlass/include"
            "${SHMEM_HOME_PATH}/shmem/include"
            "${SHMEM_HOME_PATH}/shmem/src"
            "${SHMEM_HOME_PATH}/memfabric_hybrid/include/smem/host"
            ${ASCEND_DEEPEP_HOST_INCLUDE_DIRS})
    target_link_directories(mega_moe_bf16_udma_stub
        PRIVATE
            ${ASCEND_DEEPEP_HOST_LINK_DIRS}
            "${SHMEM_HOME_PATH}/memfabric_hybrid/lib"
            "${SHMEM_HOME_PATH}/shmem/lib")
    target_link_options(mega_moe_bf16_udma_stub
        PRIVATE
            "SHELL:-Wl,-rpath,\$ORIGIN/../lib"
            "-Wl,-rpath,${ASCEND_HOME_PATH}/lib64"
            "-Wl,-rpath,${ASCEND_HOME_PATH}/aarch64-linux/lib64"
            "-Wl,-rpath,${SHMEM_HOME_PATH}/memfabric_hybrid/lib"
            "-Wl,-rpath,${SHMEM_HOME_PATH}/shmem/lib")
    target_link_libraries(mega_moe_bf16_udma_stub
        PRIVATE
            ascend_deepep_mega_moe_bf16_udma
            -Wl,--no-as-needed
            runtime
            stdc++
            ascendcl
            m
            tiling_api
            platform
            c_sec
            dl
            nnopbase
            pthread
            shmem
            -Wl,--as-needed)
endif()

if(ASCEND_DEEPEP_BUILD_MEGA_MOE)
    if(NOT ASCEND_DEEPEP_BISHENG_ARCH STREQUAL "a5")
        message(FATAL_ERROR "Standalone MegaMoe requires ASCEND_DEEPEP_BISHENG_ARCH=a5")
    endif()
    ascend_deepep_add_kernel_library(
        ascend_deepep_mega_moe
        TIMER_WITHOUT_BLOCK_LOCAL_INIT
        SOURCES
            kernels/mega_moe/mega_moe.cpp
        INCLUDES
            include
            kernels
            kernels/mega_moe/include
        COMPILE_DEFINITIONS
            CATLASS_ARCH=${ASCEND_DEEPEP_CATLASS_ARCH}
    )
    set_source_files_properties(
        kernels/mega_moe/mega_moe.cpp
        PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON
    )

    # W4A8-style standalone host runner: CSV parsing, ACL/SHMEM lifecycle,
    # input/output/workspace/peer allocation and kernel launch all live here.
    add_executable(mega_moe_stub kernels/mega_moe/mega_moe_stub.cpp)
    target_include_directories(mega_moe_stub
        PRIVATE
            "${CMAKE_SOURCE_DIR}"
            "${CMAKE_SOURCE_DIR}/include"
            "${CMAKE_SOURCE_DIR}/kernels"
            "${CMAKE_SOURCE_DIR}/kernels/mega_moe/include"
            "${CMAKE_SOURCE_DIR}/3rdparty/catlass/include"
            "${SHMEM_HOME_PATH}/shmem/include"
            "${SHMEM_HOME_PATH}/shmem/src"
            "${SHMEM_HOME_PATH}/memfabric_hybrid/include/smem/host"
            ${ASCEND_DEEPEP_HOST_INCLUDE_DIRS}
    )
    target_link_directories(mega_moe_stub
        PRIVATE
            ${ASCEND_DEEPEP_HOST_LINK_DIRS}
            "${SHMEM_HOME_PATH}/memfabric_hybrid/lib"
            "${SHMEM_HOME_PATH}/shmem/lib"
    )
    target_link_options(mega_moe_stub
        PRIVATE
            "SHELL:-Wl,-rpath,\\$ORIGIN/../lib"
            "-Wl,-rpath,${ASCEND_HOME_PATH}/lib64"
            "-Wl,-rpath,${ASCEND_HOME_PATH}/aarch64-linux/lib64"
            "-Wl,-rpath,${SHMEM_HOME_PATH}/memfabric_hybrid/lib"
            "-Wl,-rpath,${SHMEM_HOME_PATH}/shmem/lib"
    )
    target_link_libraries(mega_moe_stub
        PRIVATE
            ascend_deepep_mega_moe
            -Wl,--no-as-needed
            runtime
            stdc++
            ascendcl
            m
            tiling_api
            platform
            c_sec
            dl
            nnopbase
            pthread
            shmem
            -Wl,--as-needed
    )
endif()

if(ASCEND_DEEPEP_BUILD_MEGA_MOE_W4A8)
    if(NOT ASCEND_DEEPEP_BISHENG_ARCH STREQUAL "a5")
        message(FATAL_ERROR "Standalone MegaMoe W4A8 requires ASCEND_DEEPEP_BISHENG_ARCH=a5")
    endif()
    ascend_deepep_add_kernel_library(
        ascend_deepep_mega_moe_w4a8
        SOURCES
            kernels/mega_moe_w4a8/mega_moe_w4a8.cpp
        INCLUDES
            include
            kernels
            kernels/mega_moe_w4a8/include
        COMPILE_DEFINITIONS
            CATLASS_ARCH=${ASCEND_DEEPEP_CATLASS_ARCH}
    )
    target_compile_options(
        ascend_deepep_mega_moe_w4a8 PRIVATE --cce-auto-sync=off)
    set_source_files_properties(
        kernels/mega_moe_w4a8/mega_moe_w4a8.cpp
        PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)

    add_executable(
        mega_moe_w4a8_stub
        kernels/mega_moe_w4a8/mega_moe_w4a8_stub.cpp)
    target_include_directories(mega_moe_w4a8_stub
        PRIVATE
            "${CMAKE_SOURCE_DIR}"
            "${CMAKE_SOURCE_DIR}/include"
            "${CMAKE_SOURCE_DIR}/kernels"
            "${CMAKE_SOURCE_DIR}/kernels/mega_moe_w4a8/include"
            "${CMAKE_SOURCE_DIR}/3rdparty/catlass/include"
            "${SHMEM_HOME_PATH}/shmem/include"
            "${SHMEM_HOME_PATH}/shmem/src"
            "${SHMEM_HOME_PATH}/memfabric_hybrid/include/smem/host"
            ${ASCEND_DEEPEP_HOST_INCLUDE_DIRS})
    target_link_directories(mega_moe_w4a8_stub
        PRIVATE
            ${ASCEND_DEEPEP_HOST_LINK_DIRS}
            "${SHMEM_HOME_PATH}/memfabric_hybrid/lib"
            "${SHMEM_HOME_PATH}/shmem/lib")
    target_link_options(mega_moe_w4a8_stub
        PRIVATE
            "SHELL:-Wl,-rpath,\\$ORIGIN/../lib"
            "-Wl,-rpath,${ASCEND_HOME_PATH}/lib64"
            "-Wl,-rpath,${ASCEND_HOME_PATH}/aarch64-linux/lib64"
            "-Wl,-rpath,${SHMEM_HOME_PATH}/memfabric_hybrid/lib"
            "-Wl,-rpath,${SHMEM_HOME_PATH}/shmem/lib")
    target_link_libraries(mega_moe_w4a8_stub
        PRIVATE
            ascend_deepep_mega_moe_w4a8
            -Wl,--no-as-needed
            runtime
            stdc++
            ascendcl
            m
            tiling_api
            platform
            c_sec
            dl
            nnopbase
            pthread
            shmem
            -Wl,--as-needed)
endif()

if(ASCEND_DEEPEP_BUILD_MEGA_MOE_BF16)
    message(STATUS "MegaMoe BF16 kernel library: ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libascend_deepep_mega_moe_bf16.so")
endif()
if(ASCEND_DEEPEP_BUILD_MEGA_MOE_BF16_MGROUP)
    message(STATUS "MegaMoe BF16 MGroup kernel library: ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libascend_deepep_mega_moe_bf16_mgroup.so")
endif()

if(ASCEND_DEEPEP_BUILD_MEGA_MOE_W4A8)
    message(STATUS "MegaMoe W4A8 kernel library: ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libascend_deepep_mega_moe_w4a8.so")
endif()

message(STATUS "ascend-deepep kernel library: ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libascend_deepep_kernels.so")
if(ASCEND_DEEPEP_BUILD_MEGA_MOE)
    message(STATUS "MegaMoe kernel library: ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libascend_deepep_mega_moe.so")
endif()
message(STATUS "SHMEM installation: ${SHMEM_HOME_PATH}")

# Staged-replica helpers; independent ABI, same CANN/SHMEM as BF16 baseline.
ascend_deepep_add_kernel_library(ascend_deepep_replica
    SOURCES kernels/mega_moe_bf16/mega_moe_replica.cpp kernels/mega_moe_bf16/mega_moe_replica_planner.cpp
    INCLUDES include kernels/mega_moe_bf16/include
        "${SHMEM_HOME_PATH}/shmem/include" "${SHMEM_HOME_PATH}/shmem/src")

# Independent optimized online planner; preserve the S4/S5 helper for controls.
ascend_deepep_add_kernel_library(ascend_deepep_replica_control
    SOURCES kernels/mega_moe_bf16/mega_moe_replica_planner_control.cpp
    INCLUDES include kernels/mega_moe_bf16/include
        "${SHMEM_HOME_PATH}/shmem/include" "${SHMEM_HOME_PATH}/shmem/src")

if(ASCEND_DEEPEP_BUILD_MEGA_MOE_BF16)
    ascend_deepep_add_kernel_library(ascend_deepep_megaflux_bf16
        TIMER_WITHOUT_BLOCK_LOCAL_INIT
        SOURCES kernels/mega_moe_bf16/megaflux_bf16.cpp
                kernels/mega_moe_bf16/megaflux_bf16_backward.cpp
        INCLUDES include kernels kernels/mega_moe_bf16/include kernels/mega_moe/include
            "${SHMEM_HOME_PATH}/shmem/include" "${SHMEM_HOME_PATH}/shmem/src"
        COMPILE_DEFINITIONS CATLASS_ARCH=${ASCEND_DEEPEP_CATLASS_ARCH}
            ${BF16_GMM1_SCHEDULE_DEFINITIONS})
    set_source_files_properties(kernels/mega_moe_bf16/megaflux_bf16.cpp
        kernels/mega_moe_bf16/megaflux_bf16_backward.cpp
        PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

# Device-controlled candidate: separate symbols and SOs keep measured v2 intact.
set(MEGAFLUX_MOONEP_DIRECT_PERMILLE 0 CACHE STRING "Experimental EP8 owner-route threshold (0 disables; 1050 means max/mean<=1.05)")
if(NOT "${MEGAFLUX_MOONEP_DIRECT_PERMILLE}" MATCHES "^[0-9]+$")
    message(FATAL_ERROR "MoonEP direct threshold must be an integer")
endif()
if(NOT MEGAFLUX_MOONEP_DIRECT_PERMILLE EQUAL 0 AND
   (MEGAFLUX_MOONEP_DIRECT_PERMILLE LESS 1000 OR MEGAFLUX_MOONEP_DIRECT_PERMILLE GREATER 8000))
    message(FATAL_ERROR "MoonEP direct threshold must be 0 or 1000..8000")
endif()
option(MEGAFLUX_MOONEP_TIMER "Enable isolated MoonEP coordinate timer (requires extended header)" OFF)
if(MEGAFLUX_MOONEP_TIMER AND NOT ASCEND_DEEPEP_ENABLE_TIMER)
    message(FATAL_ERROR "MEGAFLUX_MOONEP_TIMER requires ASCEND_DEEPEP_ENABLE_TIMER=ON")
endif()
ascend_deepep_add_kernel_library(ascend_deepep_moonep_planner
    TIMER_WITHOUT_BLOCK_LOCAL_INIT
    SOURCES kernels/mega_moe_bf16/mega_moe_moonep_planner.cpp
    COMPILE_DEFINITIONS MEGAFLUX_MOONEP_DIRECT_PERMILLE=${MEGAFLUX_MOONEP_DIRECT_PERMILLE}
        MEGAFLUX_MOONEP_TIMER=$<BOOL:${MEGAFLUX_MOONEP_TIMER}>
    INCLUDES include kernels/mega_moe_bf16/include
        "${SHMEM_HOME_PATH}/shmem/include" "${SHMEM_HOME_PATH}/shmem/src")
ascend_deepep_add_kernel_library(ascend_deepep_plan_validator
    TIMER_WITHOUT_BLOCK_LOCAL_INIT
    SOURCES kernels/mega_moe_bf16/mega_moe_plan_validator.cpp
    INCLUDES include "${SHMEM_HOME_PATH}/shmem/include" "${SHMEM_HOME_PATH}/shmem/src")
ascend_deepep_add_kernel_library(ascend_deepep_unified_planner
    TIMER_WITHOUT_BLOCK_LOCAL_INIT
    SOURCES kernels/mega_moe_bf16/mega_moe_unified_planner.cpp
    INCLUDES include "${SHMEM_HOME_PATH}/shmem/include" "${SHMEM_HOME_PATH}/shmem/src")
ascend_deepep_add_kernel_library(ascend_deepep_link_calibration
    TIMER_WITHOUT_BLOCK_LOCAL_INIT
    SOURCES kernels/mega_moe_bf16/mega_moe_link_calibration.cpp
    INCLUDES include "${SHMEM_HOME_PATH}/shmem/include" "${SHMEM_HOME_PATH}/shmem/src")
ascend_deepep_add_kernel_library(ascend_deepep_step_time_planner
    TIMER_WITHOUT_BLOCK_LOCAL_INIT
    SOURCES kernels/mega_moe_bf16/mega_moe_step_time_planner.cpp
    INCLUDES include "${SHMEM_HOME_PATH}/shmem/include" "${SHMEM_HOME_PATH}/shmem/src")
ascend_deepep_add_kernel_library(ascend_deepep_large_ep_planner
    TIMER_WITHOUT_BLOCK_LOCAL_INIT
    SOURCES kernels/mega_moe_bf16/mega_moe_large_ep_planner.cpp
    COMPILE_DEFINITIONS MEGAFLUX_MOONEP_TIMER=$<BOOL:${MEGAFLUX_MOONEP_TIMER}>
    INCLUDES include kernels/mega_moe_bf16/include
        "${SHMEM_HOME_PATH}/shmem/include" "${SHMEM_HOME_PATH}/shmem/src")
ascend_deepep_add_kernel_library(ascend_deepep_replica_optimized
    SOURCES kernels/mega_moe_bf16/mega_moe_replica_planner_optimized.cpp
    INCLUDES include kernels/mega_moe_bf16/include
        "${SHMEM_HOME_PATH}/shmem/include" "${SHMEM_HOME_PATH}/shmem/src")
ascend_deepep_add_kernel_library(ascend_deepep_replica_device
    SOURCES kernels/mega_moe_bf16/mega_moe_replica_planner_device.cpp
    INCLUDES include kernels/mega_moe_bf16/include
        "${SHMEM_HOME_PATH}/shmem/include" "${SHMEM_HOME_PATH}/shmem/src")
if(ASCEND_DEEPEP_BUILD_MEGA_MOE_BF16)
    ascend_deepep_add_kernel_library(ascend_deepep_megaflux_device_bf16
        TIMER_WITHOUT_BLOCK_LOCAL_INIT
        SOURCES kernels/mega_moe_bf16/megaflux_device_bf16.cpp
                kernels/mega_moe_bf16/megaflux_device_bf16_backward.cpp
        INCLUDES include kernels kernels/mega_moe_bf16/include kernels/mega_moe/include
            "${SHMEM_HOME_PATH}/shmem/include" "${SHMEM_HOME_PATH}/shmem/src"
        COMPILE_DEFINITIONS CATLASS_ARCH=${ASCEND_DEEPEP_CATLASS_ARCH}
            ${BF16_GMM1_SCHEDULE_DEFINITIONS})
    set_source_files_properties(kernels/mega_moe_bf16/megaflux_device_bf16.cpp
        kernels/mega_moe_bf16/megaflux_device_bf16_backward.cpp
        PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

# S6 forward only; device planner/backward and all measured SOs remain controls.
if(ASCEND_DEEPEP_BUILD_MEGA_MOE_BF16)
    ascend_deepep_add_kernel_library(ascend_deepep_megaflux_pipeline_bf16
        TIMER_WITHOUT_BLOCK_LOCAL_INIT
        SOURCES kernels/mega_moe_bf16/megaflux_pipeline_bf16.cpp
        INCLUDES include kernels kernels/mega_moe_bf16/include kernels/mega_moe/include
            "${SHMEM_HOME_PATH}/shmem/include" "${SHMEM_HOME_PATH}/shmem/src"
        COMPILE_DEFINITIONS CATLASS_ARCH=${ASCEND_DEEPEP_CATLASS_ARCH}
            ${BF16_GMM1_SCHEDULE_DEFINITIONS})
    set_source_files_properties(kernels/mega_moe_bf16/megaflux_pipeline_bf16.cpp
        PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

# S6 W1 stripes; device planner/backward and all measured SOs remain controls.
if(ASCEND_DEEPEP_BUILD_MEGA_MOE_BF16)
    ascend_deepep_add_kernel_library(ascend_deepep_megaflux_stripe_bf16
        TIMER_WITHOUT_BLOCK_LOCAL_INIT
        SOURCES kernels/mega_moe_bf16/megaflux_stripe_bf16.cpp
        INCLUDES include kernels kernels/mega_moe_bf16/include kernels/mega_moe/include
            "${SHMEM_HOME_PATH}/shmem/include" "${SHMEM_HOME_PATH}/shmem/src"
        COMPILE_DEFINITIONS CATLASS_ARCH=${ASCEND_DEEPEP_CATLASS_ARCH}
            ${BF16_GMM1_SCHEDULE_DEFINITIONS})
    set_source_files_properties(kernels/mega_moe_bf16/megaflux_stripe_bf16.cpp
        PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

# Complete F+B candidate; all prior native controls remain separate targets.
if(ASCEND_DEEPEP_BUILD_MEGA_MOE_BF16)
    ascend_deepep_add_kernel_library(ascend_deepep_megaflux_full_bf16
        TIMER_WITHOUT_BLOCK_LOCAL_INIT
        SOURCES kernels/mega_moe_bf16/megaflux_full_bf16.cpp
                kernels/mega_moe_bf16/megaflux_full_bf16_backward.cpp
        INCLUDES include kernels kernels/mega_moe_bf16/include kernels/mega_moe/include
            "${SHMEM_HOME_PATH}/shmem/include" "${SHMEM_HOME_PATH}/shmem/src"
        COMPILE_DEFINITIONS CATLASS_ARCH=${ASCEND_DEEPEP_CATLASS_ARCH}
            ${BF16_GMM1_SCHEDULE_DEFINITIONS})
    set_source_files_properties(kernels/mega_moe_bf16/megaflux_full_bf16.cpp
        kernels/mega_moe_bf16/megaflux_full_bf16_backward.cpp
        PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

# S10 independent bounded W2 prefix prefetch; old backward for attribution.
if(ASCEND_DEEPEP_BUILD_MEGA_MOE_BF16)
    ascend_deepep_add_kernel_library(ascend_deepep_megaflux_prefetch_bf16
        TIMER_WITHOUT_BLOCK_LOCAL_INIT
        SOURCES kernels/mega_moe_bf16/megaflux_prefetch_bf16.cpp
                kernels/mega_moe_bf16/megaflux_prefetch_bf16_backward.cpp
        INCLUDES include kernels kernels/mega_moe_bf16/include kernels/mega_moe/include
            "${SHMEM_HOME_PATH}/shmem/include" "${SHMEM_HOME_PATH}/shmem/src"
        COMPILE_DEFINITIONS CATLASS_ARCH=${ASCEND_DEEPEP_CATLASS_ARCH}
            ${BF16_GMM1_SCHEDULE_DEFINITIONS})
    set_source_files_properties(kernels/mega_moe_bf16/megaflux_prefetch_bf16.cpp
        kernels/mega_moe_bf16/megaflux_prefetch_bf16_backward.cpp
        PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

# Bounded detail instrumentation, isolated from all production targets/ABIs.
if(ASCEND_DEEPEP_BUILD_MEGA_MOE_BF16)
    ascend_deepep_add_kernel_library(ascend_deepep_megaflux_detail_bf16
        TIMER_WITHOUT_BLOCK_LOCAL_INIT
        SOURCES kernels/mega_moe_bf16/megaflux_detail_bf16.cpp
                kernels/mega_moe_bf16/megaflux_detail_bf16_backward.cpp
        INCLUDES include kernels kernels/mega_moe_bf16/include kernels/mega_moe/include
            "${SHMEM_HOME_PATH}/shmem/include" "${SHMEM_HOME_PATH}/shmem/src"
        COMPILE_DEFINITIONS CATLASS_ARCH=${ASCEND_DEEPEP_CATLASS_ARCH}
            ${BF16_GMM1_SCHEDULE_DEFINITIONS})
    set_source_files_properties(kernels/mega_moe_bf16/megaflux_detail_bf16.cpp
        kernels/mega_moe_bf16/megaflux_detail_bf16_backward.cpp
        PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

# AscendTimerV2 coordinate stages, independent diagnostic SO.
if(ASCEND_DEEPEP_BUILD_MEGA_MOE_BF16)
    ascend_deepep_add_kernel_library(ascend_deepep_megaflux_stage_timer_bf16
        TIMER_WITHOUT_BLOCK_LOCAL_INIT
        SOURCES kernels/mega_moe_bf16/megaflux_stage_timer_bf16.cpp
                kernels/mega_moe_bf16/megaflux_stage_timer_bf16_backward.cpp
        INCLUDES include kernels kernels/mega_moe_bf16/include kernels/mega_moe/include
            "${SHMEM_HOME_PATH}/shmem/include" "${SHMEM_HOME_PATH}/shmem/src"
        COMPILE_DEFINITIONS CATLASS_ARCH=${ASCEND_DEEPEP_CATLASS_ARCH}
            ${BF16_GMM1_SCHEDULE_DEFINITIONS})
    set_source_files_properties(kernels/mega_moe_bf16/megaflux_stage_timer_bf16.cpp
        kernels/mega_moe_bf16/megaflux_stage_timer_bf16_backward.cpp
        PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

# S9 independent early owner-reduction executor.
if(ASCEND_DEEPEP_BUILD_MEGA_MOE_BF16)
    ascend_deepep_add_kernel_library(ascend_deepep_megaflux_early_bf16
        TIMER_WITHOUT_BLOCK_LOCAL_INIT
        SOURCES kernels/mega_moe_bf16/megaflux_early_bf16.cpp
                kernels/mega_moe_bf16/megaflux_early_bf16_backward.cpp
        INCLUDES include kernels kernels/mega_moe_bf16/include kernels/mega_moe/include
            "${SHMEM_HOME_PATH}/shmem/include" "${SHMEM_HOME_PATH}/shmem/src"
        COMPILE_DEFINITIONS CATLASS_ARCH=${ASCEND_DEEPEP_CATLASS_ARCH}
            ${BF16_GMM1_SCHEDULE_DEFINITIONS})
    set_source_files_properties(kernels/mega_moe_bf16/megaflux_early_bf16.cpp
        kernels/mega_moe_bf16/megaflux_early_bf16_backward.cpp
        PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

# Optional owner MTE2/Vector/MTE3 pipeline; accepted S9 stays a control.
if(ASCEND_DEEPEP_BUILD_MEGA_MOE_BF16)
    ascend_deepep_add_kernel_library(ascend_deepep_megaflux_owner_db_bf16
        TIMER_WITHOUT_BLOCK_LOCAL_INIT
        SOURCES kernels/mega_moe_bf16/megaflux_owner_db_bf16.cpp
                kernels/mega_moe_bf16/megaflux_owner_db_bf16_backward.cpp
        INCLUDES include kernels kernels/mega_moe_bf16/include kernels/mega_moe/include
            "${SHMEM_HOME_PATH}/shmem/include" "${SHMEM_HOME_PATH}/shmem/src"
        COMPILE_DEFINITIONS CATLASS_ARCH=${ASCEND_DEEPEP_CATLASS_ARCH}
            ${BF16_GMM1_SCHEDULE_DEFINITIONS})
    set_source_files_properties(kernels/mega_moe_bf16/megaflux_owner_db_bf16.cpp
        kernels/mega_moe_bf16/megaflux_owner_db_bf16_backward.cpp
        PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

# Explicit single-operator variants. Separate build directories/SOs are required;
# neither the accepted early target nor historical owner-db target gets flags.
if(ASCEND_DEEPEP_BUILD_MEGA_MOE_BF16)
    set(MEGAFLUX_OPT_DB 0 CACHE STRING "16-row owner ping-pong (0/1)")
    set(MEGAFLUX_OPT_RATIO 1 CACHE STRING "Early AIV1:late AIV0 owner task ratio (1/2/3)")
    set(MEGAFLUX_OPT_SCAN 0 CACHE STRING "Owned lookahead tasks (0/8)")
    set(MEGAFLUX_OPT_COPY 0 CACHE STRING "Weight copy ping-pong (0/1)")
    set(MEGAFLUX_OPT_CACHE 0 CACHE STRING "Per-forward AIC acquired-ready cache (0/1)")
    set(MEGAFLUX_OPT_TIMER 0 CACHE STRING "AscendTimerV2 only (0/1)")
    foreach(flag DB COPY CACHE TIMER)
        if(NOT "${MEGAFLUX_OPT_${flag}}" MATCHES "^[01]$")
            message(FATAL_ERROR "MEGAFLUX_OPT_${flag} must be 0/1")
        endif()
    endforeach()
    if(NOT MEGAFLUX_OPT_RATIO MATCHES "^[123]$" OR NOT MEGAFLUX_OPT_SCAN MATCHES "^(0|8)$")
        message(FATAL_ERROR "owner ratio must be 1/2/3 and scan must be 0/8")
    endif()
    ascend_deepep_add_kernel_library(ascend_deepep_megaflux_operator_opt_bf16
        TIMER_WITHOUT_BLOCK_LOCAL_INIT
        SOURCES kernels/mega_moe_bf16/megaflux_operator_opt_bf16.cpp
                kernels/mega_moe_bf16/megaflux_operator_opt_bf16_backward.cpp
        INCLUDES include kernels kernels/mega_moe_bf16/include kernels/mega_moe/include
            "${SHMEM_HOME_PATH}/shmem/include" "${SHMEM_HOME_PATH}/shmem/src"
        COMPILE_DEFINITIONS CATLASS_ARCH=${ASCEND_DEEPEP_CATLASS_ARCH}
            ${BF16_GMM1_SCHEDULE_DEFINITIONS}
            MEGAFLUX_OPT_DB=${MEGAFLUX_OPT_DB} DEEPEP_BF16_OWNER_RATIO=${MEGAFLUX_OPT_RATIO}
            DEEPEP_BF16_OWNER_SCAN=${MEGAFLUX_OPT_SCAN} MEGAFLUX_OPT_COPY=${MEGAFLUX_OPT_COPY}
            MEGAFLUX_OPT_CACHE=${MEGAFLUX_OPT_CACHE} MEGAFLUX_OPT_TIMER=${MEGAFLUX_OPT_TIMER})
    set_source_files_properties(kernels/mega_moe_bf16/megaflux_operator_opt_bf16.cpp
        kernels/mega_moe_bf16/megaflux_operator_opt_bf16_backward.cpp
        PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

# S6 stream modes; device planner/backward and all measured SOs remain controls.
if(ASCEND_DEEPEP_BUILD_MEGA_MOE_BF16)
    ascend_deepep_add_kernel_library(ascend_deepep_megaflux_stream_bf16
        TIMER_WITHOUT_BLOCK_LOCAL_INIT
        SOURCES kernels/mega_moe_bf16/megaflux_stream_bf16.cpp
        INCLUDES include kernels kernels/mega_moe_bf16/include kernels/mega_moe/include
            "${SHMEM_HOME_PATH}/shmem/include" "${SHMEM_HOME_PATH}/shmem/src"
        COMPILE_DEFINITIONS CATLASS_ARCH=${ASCEND_DEEPEP_CATLASS_ARCH}
            ${BF16_GMM1_SCHEDULE_DEFINITIONS})
    set_source_files_properties(kernels/mega_moe_bf16/megaflux_stream_bf16.cpp
        PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

# Private v3 worker executor. Build in a fresh directory for each flag pair.
if(ASCEND_DEEPEP_BUILD_MEGA_MOE_BF16)
    set(MEGAFLUX_WORKER_FORWARD 0 CACHE STRING "Compact forward task workers (0/1)")
    set(MEGAFLUX_BACKWARD_RESTORE 0 CACHE STRING "Integration backward weight restore (0=staged,1=expert,2=panel)")
    set(MEGAFLUX_RESTORE_OPERAND_READY 1 CACHE STRING "Wait restored panels at actual B GM-to-L1 loads (0/1)")
    set(MEGAFLUX_RESTORE_PANEL_ACQUIRE_CACHE 0 CACHE STRING "Cache completed DX panel acquires within one AIC expert call (0/1)")
    set(MEGAFLUX_RESTORE_P1_PANEL_ACQUIRE_CACHE 0 CACHE STRING "Cache completed P1 W2 acquires within one AIC expert call (0/1; experimental)")
    if(NOT "${MEGAFLUX_RESTORE_P1_PANEL_ACQUIRE_CACHE}" MATCHES "^[01]$")
        message(FATAL_ERROR "MEGAFLUX_RESTORE_P1_PANEL_ACQUIRE_CACHE must be 0 or 1")
    endif()
    if(NOT "${MEGAFLUX_RESTORE_PANEL_ACQUIRE_CACHE}" MATCHES "^[01]$")
        message(FATAL_ERROR "MEGAFLUX_RESTORE_PANEL_ACQUIRE_CACHE must be 0 or 1")
    endif()
    if(MEGAFLUX_RESTORE_PANEL_ACQUIRE_CACHE AND
       (NOT "${MEGAFLUX_BACKWARD_RESTORE}" STREQUAL "2" OR
        NOT "${MEGAFLUX_RESTORE_OPERAND_READY}" STREQUAL "1"))
        message(FATAL_ERROR "Panel acquire cache requires panel restore2 and operand-ready1")
    endif()
    set(MEGAFLUX_RESTORE_DELAY_CYCLES 0 CACHE STRING "Diagnostic delay before later weight panels; never for performance")
    set(MEGAFLUX_X_REDISPATCH_DELAY_SOURCE_CYCLES 0 CACHE STRING "Diagnostic rank0 X source-publish delay; never for performance")
    set(MEGAFLUX_X_REDISPATCH_DELAY_PULL_CYCLES 0 CACHE STRING "Diagnostic rank0 X pull-producer delay; never for performance")
    foreach(phase SOURCE PULL)
        if(NOT "${MEGAFLUX_X_REDISPATCH_DELAY_${phase}_CYCLES}" MATCHES "^[0-9]+$")
            message(FATAL_ERROR "X redispatch diagnostic delays must be nonnegative integer cycle counts")
        endif()
    endforeach()
    if(NOT "${MEGAFLUX_RESTORE_OPERAND_READY}" MATCHES "^[01]$" OR
       NOT "${MEGAFLUX_RESTORE_DELAY_CYCLES}" MATCHES "^[0-9]+$")
        message(FATAL_ERROR "Invalid restore operand readiness/delay")
    endif()
    if(NOT "${MEGAFLUX_BACKWARD_RESTORE}" MATCHES "^[012]$")
        message(FATAL_ERROR "Backward restore mode must be 0, 1 or 2")
    endif()
    set(MEGAFLUX_WORKER_BACKWARD 0 CACHE STRING "Cooperative expert backward (0/1)")
    set(MEGAFLUX_GRADIENT_PRIORITY 0 CACHE STRING "Matched producer/owner contributor priority (0/1)")
    set(MEGAFLUX_GRADIENT_KEY_CACHE 0 CACHE STRING "Compute exact gradient-order keys in one plan-table pass (0/1)")
    if(NOT "${MEGAFLUX_GRADIENT_KEY_CACHE}" MATCHES "^[01]$")
        message(FATAL_ERROR "MEGAFLUX_GRADIENT_KEY_CACHE must be 0 or 1")
    endif()
    set(MEGAFLUX_OWNER_LATE_ONLY 0 CACHE STRING "Diagnostic owner service after dX only (0/1)")
    set(MEGAFLUX_REDISPATCH_P1_RUNAHEAD 0 CACHE STRING "Diagnostic dHp runahead: 0 legacy, 1 original order, 2 home-first, 3 early-W2, 4 asymmetric P1, 5 include home-only")
    set(MEGAFLUX_REDISPATCH_TRANSPORT_BATCH 0 CACHE STRING "Routed BF16 GM transport: 0 legacy, 1 bounded static ring, 2 temporary whole-UB ring")
    option(MEGAFLUX_COORDINATE_TIMER "Enable coordinate records in worker TimerON builds" ON)
    if(NOT "${MEGAFLUX_REDISPATCH_TRANSPORT_BATCH}" MATCHES "^[012]$")
        message(FATAL_ERROR "MEGAFLUX_REDISPATCH_TRANSPORT_BATCH must be 0, 1 or 2")
    endif()
    if(NOT "${MEGAFLUX_REDISPATCH_P1_RUNAHEAD}" MATCHES "^[012345]$")
        message(FATAL_ERROR "MEGAFLUX_REDISPATCH_P1_RUNAHEAD must be 0 through 5")
    endif()
    if(NOT "${MEGAFLUX_OWNER_LATE_ONLY}" MATCHES "^[01]$")
        message(FATAL_ERROR "MEGAFLUX_OWNER_LATE_ONLY must be 0 or 1")
    endif()
    set(MEGAFLUX_EARLY_DW1_WIDE 0 CACHE STRING "Enable 64-expert early DW1 mask (0/1; experimental)")
    if(NOT "${MEGAFLUX_EARLY_DW1_WIDE}" MATCHES "^[01]$")
        message(FATAL_ERROR "MEGAFLUX_EARLY_DW1_WIDE must be 0 or 1")
    endif()
    set(MEGAFLUX_S9_SWIGLU_OVERRIDE "auto" CACHE STRING "Research S9 mode override: auto, 0, 1 or 2")
    if(NOT "${MEGAFLUX_GRADIENT_PRIORITY}" MATCHES "^[01]$")
        message(FATAL_ERROR "MEGAFLUX_GRADIENT_PRIORITY must be 0 or 1")
    endif()
    set(MEGAFLUX_BACKWARD_NONE_GM 0 CACHE STRING "Opt-in S9 save=none GM backward capability (0/1)")
    if(NOT "${MEGAFLUX_BACKWARD_NONE_GM}" MATCHES "^[01]$")
        message(FATAL_ERROR "MEGAFLUX_BACKWARD_NONE_GM must be 0 or 1")
    endif()
    if(MEGAFLUX_BACKWARD_NONE_GM AND
       (NOT "${MEGAFLUX_WORKER_FORWARD}" STREQUAL "0" OR
        NOT "${MEGAFLUX_WORKER_BACKWARD}" STREQUAL "0"))
        message(FATAL_ERROR "None GM backward requires the standalone S9 worker flags0 executor")
    endif()
    set(MEGAFLUX_SHARED_EXPERT_BACKWARD 0 CACHE STRING "Opt-in shared expert backward ABI1 (0/1)")
    if(NOT "${MEGAFLUX_SHARED_EXPERT_BACKWARD}" MATCHES "^[01]$")
        message(FATAL_ERROR "MEGAFLUX_SHARED_EXPERT_BACKWARD must be 0 or 1")
    endif()
    set(MEGAFLUX_SHARED_EXPERT_FORWARD 0 CACHE STRING "Build the opt-in local shared-expert forward ABI (0/1)")
    if(NOT "${MEGAFLUX_SHARED_EXPERT_FORWARD}" MATCHES "^[01]$")
        message(FATAL_ERROR "MEGAFLUX_SHARED_EXPERT_FORWARD must be 0 or 1")
    endif()
    if(MEGAFLUX_SHARED_EXPERT_FORWARD AND
       (NOT "${MEGAFLUX_WORKER_FORWARD}" STREQUAL "0" OR
        NOT "${MEGAFLUX_WORKER_BACKWARD}" STREQUAL "0"))
        message(FATAL_ERROR "Shared-expert forward requires the standalone worker flags0 executor")
    endif()
    # Enable the S9 panel path by default without changing staged/native builds.
    # An explicit cache value (including 0 for rollback) always wins.
    set(MEGAFLUX_REPLICATED_DW1_DEFAULT 0)
    if("${MEGAFLUX_BACKWARD_RESTORE}" STREQUAL "2")
        set(MEGAFLUX_REPLICATED_DW1_DEFAULT 1)
    endif()
    set(MEGAFLUX_BACKWARD_REPLICATED_DW1_EARLY ${MEGAFLUX_REPLICATED_DW1_DEFAULT} CACHE STRING "Replicated-expert DW1 before DX (0/1; panel default 1)")
    if(NOT "${MEGAFLUX_BACKWARD_REPLICATED_DW1_EARLY}" MATCHES "^[01]$")
        message(FATAL_ERROR "MEGAFLUX_BACKWARD_REPLICATED_DW1_EARLY must be 0 or 1")
    endif()
    if(NOT "${MEGAFLUX_BACKWARD_RESTORE}" STREQUAL "0" AND
       (NOT "${MEGAFLUX_WORKER_FORWARD}" STREQUAL "0" OR NOT "${MEGAFLUX_WORKER_BACKWARD}" STREQUAL "0"))
        message(FATAL_ERROR "Backward restore candidates require worker flags off")
    endif()
    set(MEGAFLUX_WORKER_SHARED_PREFIX 0 CACHE STRING "Shared prefix: 0=all-expert worker, 1=cooperative prefix, 2=producer-only prefix")
    set(MEGAFLUX_WORKER_EXTERNAL_REFRESH 0 CACHE STRING "Refresh exported external home weights in the fused forward (0/1)")
    set(MEGAFLUX_OWNER_ROWS 16 CACHE STRING "Standalone S9 owner subblock rows (16/32)")
    set(MEGAFLUX_DISPATCH_READY_LARGEST 0 CACHE STRING "Use existing single row-ready set for largest local expert (0/1)")
    if(NOT "${MEGAFLUX_DISPATCH_READY_LARGEST}" MATCHES "^[01]$")
        message(FATAL_ERROR "MEGAFLUX_DISPATCH_READY_LARGEST must be 0 or 1")
    endif()
    set(MEGAFLUX_OWNER_COMPACT 0 CACHE STRING "Cache standalone S9 contributors once per call (0/1)")
    set(MEGAFLUX_OWNER_EARLY_RATIO 1 CACHE STRING "S9 early AIV1 to late AIV0 owner task share (1/2; default 1)")
    if(NOT "${MEGAFLUX_OWNER_EARLY_RATIO}" MATCHES "^[12]$")
        message(FATAL_ERROR "MEGAFLUX_OWNER_EARLY_RATIO must be 1 or 2")
    endif()
    if(MEGAFLUX_OWNER_EARLY_RATIO EQUAL 2)
        if(NOT "${MEGAFLUX_BACKWARD_REPLICATED_DW1_EARLY}" STREQUAL "1" OR
           NOT "${MEGAFLUX_RESTORE_PANEL_ACQUIRE_CACHE}" STREQUAL "1" OR
           NOT "${MEGAFLUX_OWNER_ROWS}" STREQUAL "16" OR
           NOT "${MEGAFLUX_OWNER_COMPACT}" STREQUAL "0" OR
           NOT "${MEGAFLUX_DISPATCH_READY_LARGEST}" STREQUAL "0")
            message(FATAL_ERROR "Owner ratio2 requires isolated C1/panel-cache1/owner16/compact0/row-ready0")
        endif()
    endif()
    option(MEGAFLUX_OWNER_DOUBLE_BUFFER "S9 owner reduction double buffering (experimental)" OFF)
    set(MEGAFLUX_S9_EXPERIMENT_DEFINITIONS)
    if(MEGAFLUX_OWNER_DOUBLE_BUFFER)
        if(NOT "${MEGAFLUX_BACKWARD_RESTORE}" STREQUAL "2" OR
           NOT "${MEGAFLUX_WORKER_BACKWARD}" STREQUAL "0")
            message(FATAL_ERROR "Owner double buffering requires the standalone S9 panel worker")
        endif()
        list(APPEND MEGAFLUX_S9_EXPERIMENT_DEFINITIONS DEEPEP_BF16_OWNER_DB=1)
    endif()
    set(MEGAFLUX_OWNER_AIV1_TAIL 0 CACHE STRING "Let early AIV1 take paired AIV0 owner tiles (0/1)")
    if(NOT "${MEGAFLUX_OWNER_AIV1_TAIL}" MATCHES "^[01]$")
        message(FATAL_ERROR "MEGAFLUX_OWNER_AIV1_TAIL must be 0 or 1")
    endif()
    if(MEGAFLUX_OWNER_AIV1_TAIL AND
       (NOT "${MEGAFLUX_OWNER_EARLY_RATIO}" STREQUAL "2" OR
        NOT "${MEGAFLUX_BACKWARD_RESTORE}" STREQUAL "2" OR
        NOT "${MEGAFLUX_WORKER_FORWARD}" STREQUAL "0" OR
        NOT "${MEGAFLUX_WORKER_BACKWARD}" STREQUAL "0"))
        message(FATAL_ERROR "AIV1 tail requires the standalone S9 panel worker with owner ratio 2")
    endif()
    if(NOT "${MEGAFLUX_OWNER_COMPACT}" MATCHES "^[01]$")
        message(FATAL_ERROR "MEGAFLUX_OWNER_COMPACT must be 0 or 1")
    endif()
    if(NOT "${MEGAFLUX_OWNER_ROWS}" MATCHES "^(16|32)$")
        message(FATAL_ERROR "MEGAFLUX_OWNER_ROWS must be 16 or 32")
    endif()
    if(NOT "${MEGAFLUX_OWNER_ROWS}" STREQUAL "16")
        if(ASCEND_DEEPEP_ENABLE_TIMER OR NOT "${MEGAFLUX_WORKER_BACKWARD}" STREQUAL "0")
            message(FATAL_ERROR "32-row owner candidate requires Timer OFF and the standalone S9 backward")
        endif()
    endif()
    if(NOT "${MEGAFLUX_WORKER_EXTERNAL_REFRESH}" MATCHES "^[01]$")
        message(FATAL_ERROR "MEGAFLUX_WORKER_EXTERNAL_REFRESH must be 0 or 1")
    endif()
    if(MEGAFLUX_OWNER_LATE_ONLY AND
       (NOT "${MEGAFLUX_WORKER_FORWARD}" STREQUAL "0" OR
        NOT "${MEGAFLUX_WORKER_BACKWARD}" STREQUAL "0" OR
        NOT "${MEGAFLUX_BACKWARD_RESTORE}" STREQUAL "2" OR
        NOT "${MEGAFLUX_OWNER_ROWS}" STREQUAL "16" OR
        NOT "${MEGAFLUX_OWNER_COMPACT}" STREQUAL "0" OR
        NOT "${MEGAFLUX_OWNER_EARLY_RATIO}" STREQUAL "1" OR
        NOT "${MEGAFLUX_OWNER_AIV1_TAIL}" STREQUAL "0" OR
        NOT "${MEGAFLUX_GRADIENT_PRIORITY}" STREQUAL "0" OR
        NOT "${MEGAFLUX_EARLY_DW1_WIDE}" STREQUAL "0" OR
        MEGAFLUX_OWNER_DOUBLE_BUFFER))
        message(FATAL_ERROR "Late-only diagnostic requires isolated flags0/panel/owner16/ratio1/compact0/priority0/wide0")
    endif()
    if(MEGAFLUX_REDISPATCH_P1_RUNAHEAD AND
       (NOT "${MEGAFLUX_WORKER_FORWARD}" STREQUAL "0" OR
        NOT "${MEGAFLUX_WORKER_BACKWARD}" STREQUAL "0" OR
        NOT "${MEGAFLUX_BACKWARD_RESTORE}" STREQUAL "2" OR
        NOT "${MEGAFLUX_SHARED_EXPERT_BACKWARD}" STREQUAL "1" OR
        NOT "${MEGAFLUX_OWNER_LATE_ONLY}" STREQUAL "0" OR
        NOT "${MEGAFLUX_GRADIENT_PRIORITY}" STREQUAL "0" OR
        NOT "${MEGAFLUX_EARLY_DW1_WIDE}" STREQUAL "0"))
        message(FATAL_ERROR "dHp runahead requires isolated flags0/panel/redispatch/priority0/wide0/owner-late0")
    endif()
    if(NOT "${MEGAFLUX_WORKER_SHARED_PREFIX}" MATCHES "^[012]$")
        message(FATAL_ERROR "MEGAFLUX_WORKER_SHARED_PREFIX must be 0, 1 or 2")
    endif()
    foreach(flag FORWARD BACKWARD)
        if(NOT "${MEGAFLUX_WORKER_${flag}}" MATCHES "^[01]$")
            message(FATAL_ERROR "worker flags must be 0 or 1")
        endif()
    endforeach()
    if(MEGAFLUX_DISPATCH_READY_LARGEST)
        if(NOT "${MEGAFLUX_WORKER_FORWARD}" STREQUAL "0" OR
           NOT "${MEGAFLUX_WORKER_BACKWARD}" STREQUAL "0" OR
           NOT "${MEGAFLUX_OWNER_COMPACT}" STREQUAL "0" OR
           NOT "${MEGAFLUX_OWNER_ROWS}" STREQUAL "16")
            message(FATAL_ERROR "Experimental row-ready selector requires independent flags0/compact0/owner16")
        endif()
    endif()
    if(MEGAFLUX_BACKWARD_REPLICATED_DW1_EARLY)
        if(NOT "${MEGAFLUX_WORKER_FORWARD}" STREQUAL "0" OR
           NOT "${MEGAFLUX_WORKER_BACKWARD}" STREQUAL "0" OR
           NOT "${MEGAFLUX_WORKER_SHARED_PREFIX}" STREQUAL "0" OR
           NOT "${MEGAFLUX_WORKER_EXTERNAL_REFRESH}" STREQUAL "0" OR
           NOT "${MEGAFLUX_BACKWARD_RESTORE}" STREQUAL "2" OR
           NOT "${MEGAFLUX_RESTORE_OPERAND_READY}" STREQUAL "1" OR
           NOT "${MEGAFLUX_RESTORE_DELAY_CYCLES}" STREQUAL "0" OR
           NOT "${MEGAFLUX_OWNER_ROWS}" STREQUAL "16" OR
           NOT "${MEGAFLUX_OWNER_COMPACT}" STREQUAL "0" OR
           NOT "${MEGAFLUX_DISPATCH_READY_LARGEST}" STREQUAL "0" OR
           NOT "${MEGAFLUX_OPT_DB}" STREQUAL "0" OR
           NOT "${MEGAFLUX_OPT_SCAN}" STREQUAL "0" OR
           NOT "${MEGAFLUX_OPT_RATIO}" STREQUAL "1" OR
           NOT "${MEGAFLUX_OPT_COPY}" STREQUAL "0")
            message(FATAL_ERROR "Early replicated DW1 requires isolated flags0/panel2/operand-ready1/owner16/compact0 and all other candidates off")
        endif()
    endif()
    if(MEGAFLUX_RESTORE_P1_PANEL_ACQUIRE_CACHE AND
       (NOT "${MEGAFLUX_BACKWARD_REPLICATED_DW1_EARLY}" STREQUAL "1" OR
        NOT "${MEGAFLUX_RESTORE_PANEL_ACQUIRE_CACHE}" STREQUAL "1"))
        message(FATAL_ERROR "P1 panel acquire cache requires the isolated early-DW1 S9 path and DX panel cache1")
    endif()
    set(MEGAFLUX_SHARED_BACKWARD_SOURCES)
    if(MEGAFLUX_SHARED_EXPERT_BACKWARD)
        if(NOT "${MEGAFLUX_BACKWARD_REPLICATED_DW1_EARLY}" STREQUAL "1")
            message(FATAL_ERROR "Shared expert backward requires the isolated C1 panel schedule (ABI1 Host remains TimerOFF)")
        endif()
        list(APPEND MEGAFLUX_SHARED_BACKWARD_SOURCES
            kernels/mega_moe_bf16/megaflux_worker_bf16_backward_shared.cpp)
    endif()
    include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/AscendDeepEPS9.cmake")
    ascend_deepep_resolve_s9_swiglu_mode(
        "${ASCEND_DEEPEP_BF16_SWIGLU_DECOUPLE}"
        "${MEGAFLUX_SHARED_EXPERT_FORWARD}"
        "${ASCEND_DEEPEP_BF16_GM_SWIGLU_TILE_READY}"
        MEGAFLUX_EFFECTIVE_SWIGLU_DECOUPLE_MODE)
    if(NOT "${MEGAFLUX_S9_SWIGLU_OVERRIDE}" STREQUAL "auto")
        if(NOT "${MEGAFLUX_S9_SWIGLU_OVERRIDE}" MATCHES "^[012]$")
            message(FATAL_ERROR "MEGAFLUX_S9_SWIGLU_OVERRIDE must be auto, 0, 1 or 2")
        endif()
        set(MEGAFLUX_EFFECTIVE_SWIGLU_DECOUPLE_MODE "${MEGAFLUX_S9_SWIGLU_OVERRIDE}")
    endif()
    set(MEGAFLUX_EFFECTIVE_SWIGLU_DECOUPLE_MODE
        "${MEGAFLUX_EFFECTIVE_SWIGLU_DECOUPLE_MODE}" CACHE INTERNAL
        "Resolved S9 kernel SwiGLU mode (legacy default off)" FORCE)
    message(STATUS "MegaFlux S9 effective SwiGLU mode: ${MEGAFLUX_EFFECTIVE_SWIGLU_DECOUPLE_MODE}")
    ascend_deepep_add_kernel_library(ascend_deepep_megaflux_worker_bf16
        TIMER_WITHOUT_BLOCK_LOCAL_INIT
        SOURCES kernels/mega_moe_bf16/megaflux_worker_bf16.cpp
                kernels/mega_moe_bf16/megaflux_worker_bf16_backward.cpp
                ${MEGAFLUX_SHARED_BACKWARD_SOURCES}
        INCLUDES include kernels kernels/mega_moe_bf16/include kernels/mega_moe/include
            "${SHMEM_HOME_PATH}/shmem/include" "${SHMEM_HOME_PATH}/shmem/src"
        COMPILE_DEFINITIONS CATLASS_ARCH=${ASCEND_DEEPEP_CATLASS_ARCH}
            MEGAFLUX_WORKER_FORWARD=${MEGAFLUX_WORKER_FORWARD}
            MEGAFLUX_SHARED_EXPERT_FORWARD=${MEGAFLUX_SHARED_EXPERT_FORWARD}
            MEGAFLUX_BACKWARD_RESTORE=${MEGAFLUX_BACKWARD_RESTORE}
            MEGAFLUX_RESTORE_OPERAND_READY=${MEGAFLUX_RESTORE_OPERAND_READY}
            MEGAFLUX_RESTORE_PANEL_ACQUIRE_CACHE=${MEGAFLUX_RESTORE_PANEL_ACQUIRE_CACHE}
            MEGAFLUX_RESTORE_P1_PANEL_ACQUIRE_CACHE=${MEGAFLUX_RESTORE_P1_PANEL_ACQUIRE_CACHE}
            MEGAFLUX_RESTORE_DELAY_CYCLES=${MEGAFLUX_RESTORE_DELAY_CYCLES}
            MEGAFLUX_X_REDISPATCH_DELAY_SOURCE_CYCLES=${MEGAFLUX_X_REDISPATCH_DELAY_SOURCE_CYCLES}
            MEGAFLUX_X_REDISPATCH_DELAY_PULL_CYCLES=${MEGAFLUX_X_REDISPATCH_DELAY_PULL_CYCLES}
            MEGAFLUX_WORKER_BACKWARD=${MEGAFLUX_WORKER_BACKWARD}
            MEGAFLUX_GRADIENT_PRIORITY=${MEGAFLUX_GRADIENT_PRIORITY}
            MEGAFLUX_GRADIENT_KEY_CACHE=${MEGAFLUX_GRADIENT_KEY_CACHE}
            MEGAFLUX_OWNER_LATE_ONLY=${MEGAFLUX_OWNER_LATE_ONLY}
            MEGAFLUX_REDISPATCH_P1_RUNAHEAD=${MEGAFLUX_REDISPATCH_P1_RUNAHEAD}
            MEGAFLUX_REDISPATCH_TRANSPORT_BATCH=${MEGAFLUX_REDISPATCH_TRANSPORT_BATCH}
            MEGAFLUX_COORDINATE_TIMER=$<BOOL:${MEGAFLUX_COORDINATE_TIMER}>
            MEGAFLUX_EARLY_DW1_WIDE=${MEGAFLUX_EARLY_DW1_WIDE}
            MEGAFLUX_BACKWARD_NONE_GM=${MEGAFLUX_BACKWARD_NONE_GM}
            MEGAFLUX_SHARED_EXPERT_BACKWARD=${MEGAFLUX_SHARED_EXPERT_BACKWARD}
            MEGAFLUX_BACKWARD_REPLICATED_DW1_EARLY=${MEGAFLUX_BACKWARD_REPLICATED_DW1_EARLY}
            MEGAFLUX_WORKER_SHARED_PREFIX=${MEGAFLUX_WORKER_SHARED_PREFIX}
            MEGAFLUX_WORKER_EXTERNAL_REFRESH=${MEGAFLUX_WORKER_EXTERNAL_REFRESH}
            MEGAFLUX_OWNER_ROWS=${MEGAFLUX_OWNER_ROWS}
            MEGAFLUX_OWNER_COMPACT=${MEGAFLUX_OWNER_COMPACT}
            ${MEGAFLUX_S9_EXPERIMENT_DEFINITIONS}
            DEEPEP_BF16_OWNER_RATIO=${MEGAFLUX_OWNER_EARLY_RATIO}
            MEGAFLUX_OWNER_AIV1_TAIL=${MEGAFLUX_OWNER_AIV1_TAIL}
            MEGAFLUX_DISPATCH_READY_LARGEST=${MEGAFLUX_DISPATCH_READY_LARGEST}
            DEEPEP_BF16_SWIGLU_DECOUPLE_MODE=${BF16_SWIGLU_DECOUPLE_MODE}
            DEEPEP_BF16_SWIGLU_DECOUPLE=${MEGAFLUX_EFFECTIVE_SWIGLU_DECOUPLE_MODE}
            DEEPEP_BF16_GM_SWIGLU_TILE_READY=$<BOOL:${ASCEND_DEEPEP_BF16_GM_SWIGLU_TILE_READY}>
            ${BF16_GMM1_SCHEDULE_DEFINITIONS})
    set_source_files_properties(kernels/mega_moe_bf16/megaflux_worker_bf16.cpp
        kernels/mega_moe_bf16/megaflux_worker_bf16_backward.cpp
        kernels/mega_moe_bf16/megaflux_worker_bf16_backward_shared.cpp
        PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

option(MEGAFLUX_SUMMARY_TIMER "Phase-level original AscendTimerV2 instrumentation" ON)
option(MEGAFLUX_CALIBRATION_TIMER "Compact original producer/owner tile samples for planner calibration" OFF)
if(MEGAFLUX_CALIBRATION_TIMER)
    if(NOT ASCEND_DEEPEP_ENABLE_TIMER OR NOT MEGAFLUX_SUMMARY_TIMER)
        message(FATAL_ERROR "Calibration timer requires TimerON and summary timer")
    endif()
    add_compile_definitions(MEGAFLUX_CALIBRATION_TIMER=1)
endif()
if(MEGAFLUX_OWNER_AIV1_TAIL AND ASCEND_DEEPEP_ENABLE_TIMER AND NOT MEGAFLUX_SUMMARY_TIMER)
    message(FATAL_ERROR "AIV1 tail diagnostic requires summary TimerV2 to record both phase pairs")
endif()
if(MEGAFLUX_SUMMARY_TIMER)
    add_compile_definitions(MEGAFLUX_SUMMARY_TIMER=1)
endif()
