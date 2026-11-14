include(CMakeParseArguments)

function(ascend_deepep_add_kernel_library TARGET)
    set(OPTIONS TIMER_WITHOUT_BLOCK_LOCAL_INIT)
    set(ONE_VALUE_ARGS)
    set(MULTI_VALUE_ARGS SOURCES INCLUDES COMPILE_DEFINITIONS)
    cmake_parse_arguments(ARG "${OPTIONS}" "${ONE_VALUE_ARGS}" "${MULTI_VALUE_ARGS}" ${ARGN})

    if(NOT ARG_SOURCES)
        message(FATAL_ERROR "${TARGET}: SOURCES must not be empty")
    endif()

    execute_process(
        COMMAND "${CMAKE_CXX_COMPILER}" -v
        OUTPUT_VARIABLE BISHENG_VERSION_STDOUT
        ERROR_VARIABLE BISHENG_VERSION_STDERR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_STRIP_TRAILING_WHITESPACE
    )
    set(BISHENG_VERSION_OUTPUT "${BISHENG_VERSION_STDOUT} ${BISHENG_VERSION_STDERR}")
    string(REGEX MATCH
        "([0-9][0-9][0-9][0-9])-([0-9][0-9])-([0-9][0-9])"
        BISHENG_BUILD_DATE
        "${BISHENG_VERSION_OUTPUT}"
    )
    set(ASCEND_DEEPEP_USE_NEW_ASCENDC_FRONTEND FALSE)
    if(BISHENG_BUILD_DATE)
        string(REGEX REPLACE
            "([0-9][0-9][0-9][0-9])-([0-9][0-9])-([0-9][0-9])"
            "\\1\\2\\3"
            BISHENG_BUILD_TIMESTAMP
            "${BISHENG_BUILD_DATE}"
        )
        if(BISHENG_BUILD_TIMESTAMP GREATER_EQUAL 20260327)
            set(ASCEND_DEEPEP_USE_NEW_ASCENDC_FRONTEND TRUE)
        endif()
    endif()

    if(ASCEND_DEEPEP_BISHENG_ARCH STREQUAL "a5")
        set(CCE_AICORE_ARCH_BASE dav-c310)
        set(ASCEND_NPU_ARCH dav-3510)
        set(ASCEND_FFTS_COMPILE_FLAG)
    elseif(ASCEND_DEEPEP_BISHENG_ARCH STREQUAL "a2")
        set(CCE_AICORE_ARCH_BASE dav-c220)
        set(ASCEND_NPU_ARCH dav-2201)
        set(ASCEND_FFTS_COMPILE_FLAG --cce-auto-set-ffts=false)
    else()
        message(FATAL_ERROR "Unsupported ASCEND_DEEPEP_BISHENG_ARCH=${ASCEND_DEEPEP_BISHENG_ARCH}")
    endif()

    if(ASCEND_DEEPEP_USE_NEW_ASCENDC_FRONTEND)
        set(ASCEND_KERNEL_COMPILE_OPTIONS
            -xasc
            -Xhost-start -ftrapv -D__ASC_NPU_HOST__ -Xhost-end
            "--npu-arch=${ASCEND_NPU_ARCH}"
            --cce-auto-infer-kernel-type=false
            ${ASCEND_FFTS_COMPILE_FLAG}
        )
        set(ASCEND_KERNEL_LINK_ARCH_OPTIONS)
        message(STATUS
            "${TARGET}: using new AscendC frontend (-xasc --npu-arch=${ASCEND_NPU_ARCH}); "
            "Bisheng build date ${BISHENG_BUILD_DATE}")
    else()
        set(ASCEND_KERNEL_COMPILE_OPTIONS
            -xcce
            "--cce-aicore-arch=${CCE_AICORE_ARCH_BASE}"
            -Xhost-start -ftrapv -Xhost-end
            "SHELL:-mllvm -cce-aicore-stack-size=0x8000"
            "SHELL:-mllvm -cce-aicore-function-stack-size=0x8000"
            "SHELL:-mllvm -cce-aicore-record-overflow=true"
            "SHELL:-mllvm -cce-aicore-addr-transform"
            "SHELL:-mllvm -cce-aicore-dcci-insert-for-scalar=false"
        )
        set(ASCEND_KERNEL_LINK_ARCH_OPTIONS
            "--cce-aicore-arch=${CCE_AICORE_ARCH_BASE}"
        )
        if(BISHENG_BUILD_DATE)
            message(STATUS
                "${TARGET}: using legacy AscendC frontend "
                "(-xcce --cce-aicore-arch=${CCE_AICORE_ARCH_BASE}); "
                "Bisheng build date ${BISHENG_BUILD_DATE}")
        else()
            message(WARNING
                "${TARGET}: cannot extract a build date from `bisheng -v`; "
                "falling back to the legacy AscendC frontend")
        endif()
    endif()

    set(KERNEL_SOURCES)
    foreach(SOURCE IN LISTS ARG_SOURCES)
        if(IS_ABSOLUTE "${SOURCE}")
            list(APPEND KERNEL_SOURCES "${SOURCE}")
        else()
            get_filename_component(ABS_SOURCE "${SOURCE}" ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")
            list(APPEND KERNEL_SOURCES "${ABS_SOURCE}")
        endif()
    endforeach()

    set(KERNEL_INCLUDE_DIRS)
    foreach(INCLUDE_DIR IN LISTS ARG_INCLUDES)
        if(NOT IS_ABSOLUTE "${INCLUDE_DIR}")
            get_filename_component(INCLUDE_DIR "${INCLUDE_DIR}" ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")
        endif()
        list(APPEND KERNEL_INCLUDE_DIRS "${INCLUDE_DIR}")
    endforeach()

    set(COMPILE_DEF_FLAGS)
    foreach(DEF IN LISTS ARG_COMPILE_DEFINITIONS)
        list(APPEND COMPILE_DEF_FLAGS "-D${DEF}")
    endforeach()

    set(PROFILING_FLAGS)
    if(ASCEND_DEEPEP_ENABLE_TIMER)
        list(APPEND PROFILING_FLAGS -DENABLE_TIMER)
        if(NOT ARG_TIMER_WITHOUT_BLOCK_LOCAL_INIT)
            list(APPEND PROFILING_FLAGS --cce-aicore-block-local-init)
        endif()
    endif()

    get_filename_component(ASCEND_HOME_REAL_PATH "${ASCEND_HOME_PATH}" REALPATH)
    get_filename_component(ASCEND_BISHENG_BIN_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
    get_filename_component(ASCEND_BISHENG_COMPILER_DIR "${ASCEND_BISHENG_BIN_DIR}" DIRECTORY)
    get_filename_component(ASCEND_BISHENG_TOOLS_DIR "${ASCEND_BISHENG_COMPILER_DIR}" DIRECTORY)
    get_filename_component(ASCEND_BISHENG_ROOT_DIR "${ASCEND_BISHENG_TOOLS_DIR}" DIRECTORY)
    set(ASCEND_CANN_ROOT_CANDIDATES
        "${ASCEND_HOME_PATH}"
        "${ASCEND_HOME_REAL_PATH}"
        "${ASCEND_BISHENG_ROOT_DIR}"
    )
    set(ASCEND_CANN_EXTRA_INCLUDE_CANDIDATES)
    if(DEFINED ASCEND_DEEPEP_RUNTIME_INCLUDE_DIR)
        list(APPEND ASCEND_CANN_EXTRA_INCLUDE_CANDIDATES "${ASCEND_DEEPEP_RUNTIME_INCLUDE_DIR}")
    elseif(DEFINED ENV{ASCEND_DEEPEP_RUNTIME_INCLUDE_DIR})
        list(APPEND ASCEND_CANN_EXTRA_INCLUDE_CANDIDATES "$ENV{ASCEND_DEEPEP_RUNTIME_INCLUDE_DIR}")
    endif()
    if(DEFINED ENV{ASCEND_DRIVER_PATH})
        list(APPEND ASCEND_CANN_ROOT_CANDIDATES "$ENV{ASCEND_DRIVER_PATH}")
    endif()
    if(EXISTS "/usr/local/Ascend")
        file(GLOB ASCEND_INSTALLED_CANN_ROOTS
            LIST_DIRECTORIES TRUE
            "/usr/local/Ascend/cann-*"
        )
        list(APPEND ASCEND_CANN_ROOT_CANDIDATES ${ASCEND_INSTALLED_CANN_ROOTS})
        if(EXISTS "/usr/local/Ascend/driver")
            list(APPEND ASCEND_CANN_ROOT_CANDIDATES "/usr/local/Ascend/driver")
        endif()
        file(GLOB_RECURSE ASCEND_INSTALLED_RT_HEADERS
            LIST_DIRECTORIES FALSE
            "/usr/local/Ascend/*/rt.h"
        )
        foreach(ASCEND_INSTALLED_RT_HEADER IN LISTS ASCEND_INSTALLED_RT_HEADERS)
            get_filename_component(ASCEND_INSTALLED_RT_INCLUDE_DIR "${ASCEND_INSTALLED_RT_HEADER}" DIRECTORY)
            list(APPEND ASCEND_CANN_EXTRA_INCLUDE_CANDIDATES "${ASCEND_INSTALLED_RT_INCLUDE_DIR}")
        endforeach()
        if(UNIX)
            execute_process(
                COMMAND find /usr/local/Ascend -name rt.h -type f
                OUTPUT_VARIABLE ASCEND_FIND_RT_HEADERS
                ERROR_QUIET
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )
            if(ASCEND_FIND_RT_HEADERS)
                string(REPLACE "\n" ";" ASCEND_FIND_RT_HEADERS "${ASCEND_FIND_RT_HEADERS}")
                foreach(ASCEND_FIND_RT_HEADER IN LISTS ASCEND_FIND_RT_HEADERS)
                    get_filename_component(ASCEND_FIND_RT_INCLUDE_DIR "${ASCEND_FIND_RT_HEADER}" DIRECTORY)
                    list(APPEND ASCEND_CANN_EXTRA_INCLUDE_CANDIDATES "${ASCEND_FIND_RT_INCLUDE_DIR}")
                endforeach()
            endif()
        endif()
    endif()
    list(REMOVE_DUPLICATES ASCEND_CANN_ROOT_CANDIDATES)

    set(ASCEND_CANN_INCLUDE_CANDIDATES)
    set(ASCENDC_DEVICE_INCLUDE_CANDIDATES)
    set(ASCEND_CANN_LINK_CANDIDATES)
    foreach(ASCEND_CANN_ROOT IN LISTS ASCEND_CANN_ROOT_CANDIDATES)
        list(APPEND ASCENDC_DEVICE_INCLUDE_CANDIDATES
            "${ASCEND_CANN_ROOT}/compiler/tikcpp"
            "${ASCEND_CANN_ROOT}/compiler/tikcpp/tikcfw"
            "${ASCEND_CANN_ROOT}/compiler/tikcpp/tikcfw/impl"
            "${ASCEND_CANN_ROOT}/compiler/tikcpp/tikcfw/interface"
            "${ASCEND_CANN_ROOT}/${CMAKE_SYSTEM_PROCESSOR}-linux/asc/include"
        )
        list(APPEND ASCEND_CANN_INCLUDE_CANDIDATES
            "${ASCEND_CANN_ROOT}/include"
            "${ASCEND_CANN_ROOT}/include/aclnn"
            "${ASCEND_CANN_ROOT}/include/experiment/runtime"
            "${ASCEND_CANN_ROOT}/include/experiment/msprof"
            "${ASCEND_CANN_ROOT}/pkg_inc"
            "${ASCEND_CANN_ROOT}/pkg_inc/profiling"
            "${ASCEND_CANN_ROOT}/pkg_inc/runtime"
            "${ASCEND_CANN_ROOT}/pkg_inc/runtime/runtime"
            "${ASCEND_CANN_ROOT}/runtime/include"
            "${ASCEND_CANN_ROOT}/include/runtime"
            "${ASCEND_CANN_ROOT}/include/acl/runtime"
            "${ASCEND_CANN_ROOT}/aarch64-linux/include"
            "${ASCEND_CANN_ROOT}/aarch64-linux/pkg_inc"
            "${ASCEND_CANN_ROOT}/aarch64-linux/pkg_inc/profiling"
            "${ASCEND_CANN_ROOT}/aarch64-linux/pkg_inc/runtime"
            "${ASCEND_CANN_ROOT}/aarch64-linux/pkg_inc/runtime/runtime"
            "${ASCEND_CANN_ROOT}/aarch64-linux/runtime/include"
            "${ASCEND_CANN_ROOT}/aarch64-linux/include/runtime"
            "${ASCEND_CANN_ROOT}/arm64-linux/include"
            "${ASCEND_CANN_ROOT}/arm64-linux/pkg_inc"
            "${ASCEND_CANN_ROOT}/arm64-linux/pkg_inc/profiling"
            "${ASCEND_CANN_ROOT}/arm64-linux/pkg_inc/runtime"
            "${ASCEND_CANN_ROOT}/arm64-linux/pkg_inc/runtime/runtime"
            "${ASCEND_CANN_ROOT}/arm64-linux/runtime/include"
            "${ASCEND_CANN_ROOT}/arm64-linux/include/runtime"
            "${ASCEND_CANN_ROOT}/x86_64-linux/include"
            "${ASCEND_CANN_ROOT}/x86_64-linux/pkg_inc"
            "${ASCEND_CANN_ROOT}/x86_64-linux/pkg_inc/profiling"
            "${ASCEND_CANN_ROOT}/x86_64-linux/pkg_inc/runtime"
            "${ASCEND_CANN_ROOT}/x86_64-linux/pkg_inc/runtime/runtime"
            "${ASCEND_CANN_ROOT}/x86_64-linux/runtime/include"
            "${ASCEND_CANN_ROOT}/x86_64-linux/include/runtime"
            "${ASCEND_CANN_ROOT}/${CMAKE_SYSTEM_PROCESSOR}-linux/include"
            "${ASCEND_CANN_ROOT}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc"
            "${ASCEND_CANN_ROOT}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/profiling"
            "${ASCEND_CANN_ROOT}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/runtime"
            "${ASCEND_CANN_ROOT}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/runtime/runtime"
            "${ASCEND_CANN_ROOT}/${CMAKE_SYSTEM_PROCESSOR}-linux/runtime/include"
            "${ASCEND_CANN_ROOT}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/runtime"
        )
        list(APPEND ASCEND_CANN_LINK_CANDIDATES
            "${ASCEND_CANN_ROOT}/lib64"
            "${ASCEND_CANN_ROOT}/aarch64-linux/lib64"
            "${ASCEND_CANN_ROOT}/x86_64-linux/lib64"
            "${ASCEND_CANN_ROOT}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64"
        )
    endforeach()
    list(APPEND ASCEND_CANN_INCLUDE_CANDIDATES ${ASCEND_CANN_EXTRA_INCLUDE_CANDIDATES})
    list(REMOVE_DUPLICATES ASCENDC_DEVICE_INCLUDE_CANDIDATES)
    list(REMOVE_DUPLICATES ASCEND_CANN_INCLUDE_CANDIDATES)
    list(REMOVE_DUPLICATES ASCEND_CANN_LINK_CANDIDATES)

    find_path(ASCEND_ACL_INCLUDE_DIR
        NAMES acl/acl.h
        HINTS ${ASCEND_CANN_INCLUDE_CANDIDATES}
        NO_DEFAULT_PATH
    )
    if(NOT ASCEND_ACL_INCLUDE_DIR)
        message(FATAL_ERROR
            "Cannot find acl/acl.h under ASCEND_HOME_PATH=${ASCEND_HOME_PATH}. "
            "The include path must be the directory that contains the acl/ subdirectory, "
            "for example <cann>/include, not <cann>/include/acl.")
    endif()
    message(STATUS "ACL include root: ${ASCEND_ACL_INCLUDE_DIR}")

    find_path(ASCEND_RUNTIME_INCLUDE_DIR
        NAMES rt.h
        HINTS ${ASCEND_CANN_INCLUDE_CANDIDATES}
        NO_DEFAULT_PATH
    )
    if(NOT ASCEND_RUNTIME_INCLUDE_DIR)
        message(FATAL_ERROR
            "Cannot find rt.h under ASCEND_HOME_PATH=${ASCEND_HOME_PATH}. "
            "Bisheng runtime headers include rt.h directly, so the "
            "directory that contains rt.h must be on the compiler include path. "
            "Find it with `find <parent-of-ASCEND_HOME_PATH> -type f -name rt.h` "
            "and pass "
            "`-DASCEND_DEEPEP_RUNTIME_INCLUDE_DIR=<directory-containing-rt.h>` "
            "or use `scripts/build_mega_moe.sh --runtime-include-dir <directory>`.")
    endif()
    message(STATUS "Runtime include root: ${ASCEND_RUNTIME_INCLUDE_DIR}")

    add_library(${TARGET} SHARED ${KERNEL_SOURCES})
    set_target_properties(${TARGET}
        PROPERTIES
            UNITY_BUILD ON
            UNITY_BUILD_BATCH_SIZE 16 # 后面算子增多的情况，可以换成多个so的实现
    )
    target_compile_options(${TARGET}
        PRIVATE
            -O2
            ${ASCEND_KERNEL_COMPILE_OPTIONS}
            # "--cce-enable-print"
            ${PROFILING_FLAGS}
            ${COMPILE_DEF_FLAGS}
    )

    target_compile_definitions(${TARGET} PRIVATE L2_CACHE_HINT)
    if(ASCEND_DEEPEP_USE_NEW_ASCENDC_FRONTEND)
        target_compile_definitions(${TARGET}
            PRIVATE ASCEND_DEEPEP_USE_NEW_ASCENDC_FRONTEND=1)
    endif()
    target_include_directories(${TARGET}
        PRIVATE
            "${CMAKE_SOURCE_DIR}/include"
            "${CMAKE_SOURCE_DIR}/kernels"
            ${ASCENDC_DEVICE_INCLUDE_CANDIDATES}
            "${ASCEND_ACL_INCLUDE_DIR}"
            "${ASCEND_RUNTIME_INCLUDE_DIR}"
            "${CMAKE_SOURCE_DIR}/3rdparty/catlass/include"
            "${CMAKE_SOURCE_DIR}/3rdparty/catlass/examples/common"
            "${SHMEM_HOME_PATH}/shmem/include"
            "${SHMEM_HOME_PATH}/shmem/src"
            "${SHMEM_HOME_PATH}/shmem/src/device"
            "${SHMEM_HOME_PATH}/memfabric_hybrid/include/smem/host"
            "${SHMEM_HOME_PATH}/memfabric_hybrid/include/smem/device"
            ${ASCEND_CANN_INCLUDE_CANDIDATES}
            ${KERNEL_INCLUDE_DIRS}
    )

    target_link_directories(${TARGET}
        PRIVATE
            ${ASCEND_CANN_LINK_CANDIDATES}
            "${SHMEM_HOME_PATH}/memfabric_hybrid/lib"
            "${SHMEM_HOME_PATH}/shmem/lib"
    )
    target_link_options(${TARGET}
        PRIVATE
            --cce-fatobj-link
            ${ASCEND_KERNEL_LINK_ARCH_OPTIONS}
            "SHELL:-Wl,-rpath,\\$ORIGIN"
            "-Wl,-rpath,${ASCEND_HOME_PATH}/lib64"
            "-Wl,-rpath,${ASCEND_HOME_PATH}/aarch64-linux/lib64"
            "-Wl,-rpath,${SHMEM_HOME_PATH}/memfabric_hybrid/lib"
            "-Wl,-rpath,${SHMEM_HOME_PATH}/shmem/lib"
    )
    target_link_libraries(${TARGET}
        PRIVATE
            -Wl,--no-as-needed
            runtime
            ascendcl
            shmem
            graph_base
            tiling_api
            platform
            register
            dl
            -Wl,--as-needed
            stdc++
            m
    )

    set(${TARGET}_LIBRARY "$<TARGET_FILE:${TARGET}>" PARENT_SCOPE)
    set(ASCEND_DEEPEP_HOST_INCLUDE_DIRS
        "${ASCEND_ACL_INCLUDE_DIR};${ASCEND_RUNTIME_INCLUDE_DIR};${ASCEND_CANN_INCLUDE_CANDIDATES}"
        PARENT_SCOPE)
    set(ASCEND_DEEPEP_HOST_LINK_DIRS
        "${ASCEND_CANN_LINK_CANDIDATES}"
        PARENT_SCOPE)
endfunction()
