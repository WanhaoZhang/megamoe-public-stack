include_guard(GLOBAL)

# The legacy S9 target supplied only the unused _MODE alias, so its kernel
# resolved DEEPEP_BF16_SWIGLU_DECOUPLE to 0. Keep that default until the caller
# explicitly selects GM or opts into the new shared/tile forward path.
function(ascend_deepep_resolve_s9_swiglu_mode requested_mode shared_forward gm_tile out_var)
    if("${requested_mode}" STREQUAL "off")
        set(resolved_mode 0)
    elseif("${requested_mode}" STREQUAL "on")
        set(resolved_mode 1)
    elseif("${requested_mode}" STREQUAL "auto")
        if(shared_forward OR gm_tile)
            set(resolved_mode 2)
        else()
            set(resolved_mode 0)
        endif()
    else()
        message(FATAL_ERROR "S9 SwiGLU decoupling must be off, on or auto")
    endif()
    set(${out_var} ${resolved_mode} PARENT_SCOPE)
endfunction()
