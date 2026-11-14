include("${CMAKE_CURRENT_LIST_DIR}/AscendDeepEPS9.cmake")

# requested mode | shared forward | GM tile | effective kernel mode
set(cases
    "off|0|OFF|0" "off|0|ON|0" "off|1|OFF|0" "off|1|ON|0"
    "on|0|OFF|1" "on|0|ON|1" "on|1|OFF|1" "on|1|ON|1"
    "auto|0|OFF|0" "auto|0|ON|2" "auto|1|OFF|2" "auto|1|ON|2")
set(checked 0)
foreach(backward IN ITEMS 0 1)
    # Existing shared backward alone must not enable the new forward policy.
    set(MEGAFLUX_SHARED_EXPERT_BACKWARD ${backward})
    foreach(row IN LISTS cases)
        string(REPLACE "|" ";" fields "${row}")
        list(GET fields 0 requested)
        list(GET fields 1 shared_forward)
        list(GET fields 2 gm_tile)
        list(GET fields 3 expected)
        ascend_deepep_resolve_s9_swiglu_mode(
            "${requested}" "${shared_forward}" "${gm_tile}" actual)
        if(NOT "${actual}" STREQUAL "${expected}")
            message(FATAL_ERROR "S9 policy mismatch: backward=${backward}, ${row}, actual=${actual}")
        endif()
        math(EXPR checked "${checked} + 1")
    endforeach()
endforeach()

ascend_deepep_resolve_s9_swiglu_mode("on" "1" "ON" validated_mode)
set(validated_before
    "DEEPEP_BF16_SWIGLU_DECOUPLE_MODE=1;DEEPEP_BF16_SWIGLU_DECOUPLE=1;DEEPEP_BF16_GM_SWIGLU_TILE_READY=1")
set(validated_after
    "DEEPEP_BF16_SWIGLU_DECOUPLE_MODE=1;DEEPEP_BF16_SWIGLU_DECOUPLE=${validated_mode};DEEPEP_BF16_GM_SWIGLU_TILE_READY=1")
if(NOT "${validated_before}" STREQUAL "${validated_after}")
    message(FATAL_ERROR "Validated shared/on/tile recipe changed")
endif()
message(STATUS "S9 SwiGLU policy: ${checked} cases passed; legacy auto=0, validated shared/on/tile=1")
message(STATUS "Validated recipe definitions unchanged: ${validated_after}")
