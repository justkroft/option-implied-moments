cmake_minimum_required(VERSION 3.22)
include(CTest)

# Tests
function(add_c_test_executable test_file)
    file(RELATIVE_PATH rel "${CMAKE_CURRENT_SOURCE_DIR}" "${test_file}")
    get_filename_component(rel_dir  "${rel}" DIRECTORY)   # ext
    get_filename_component(stem     "${rel}" NAME_WE)     # test_trapezoid_core

    if(rel_dir)
        set(target_name "${rel_dir}_${stem}")             # ext_test_trapezoid_core
    else()
        set(target_name "${stem}")
    endif()

    # --- derive a human-readable CTest name
    # Strip the "test_" prefix from the stem for readability, then
    # combine with the subdirectory: ext/trapezoid_core
    string(REGEX REPLACE "^test_" "" bare_stem "${stem}")  # trapezoid_core
    if(rel_dir)
        set(ctest_name "${rel_dir}/${bare_stem}")          # ext/trapezoid_core
    else()
        set(ctest_name "${bare_stem}")
    endif()

    # --- derive path to the source under test
    # tests/ext/test_file.c  ->  src/ext/file.c
    if(rel_dir)
        set(src_file "${CMAKE_SOURCE_DIR}/src/${rel_dir}/${bare_stem}.c")
    else()
        set(src_file "${CMAKE_SOURCE_DIR}/src/${bare_stem}.c")
    endif()

    # --- build the executable
    if(EXISTS "${src_file}")
        get_filename_component(src_dir "${src_file}" DIRECTORY)
        add_executable("${target_name}" "${test_file}" "${src_file}")
    else()
        set(src_dir "")
        add_executable("${target_name}" "${test_file}")
    endif()

    target_include_directories("${target_name}" PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"   # test.h
        $<$<BOOL:${src_dir}>:${src_dir}>  # header next to the .c file, if it exists
    )

    target_compile_options("${target_name}" PRIVATE
        $<$<NOT:$<C_COMPILER_ID:MSVC>>:-O2 -w>
        $<$<C_COMPILER_ID:MSVC>:/O2 /w>
    )

    target_link_libraries("${target_name}" PRIVATE
        $<$<NOT:$<C_COMPILER_ID:MSVC>>:m>
    )

    # Coverage flags (GCC / Clang only)
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options("${target_name}" PRIVATE --coverage)
        target_link_options("${target_name}"    PRIVATE --coverage)
    endif()

    set_target_properties("${target_name}" PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/tests/${rel_dir}"
    )

    # --- register with CTest
    add_test(
        NAME    "${ctest_name}"
        COMMAND "${target_name}"
    )

endfunction()

# ---------------------------------------------------------------------------
# Auto-discover all C test files under tests/
#
# GLOB_RECURSE finds e.g.:
#   tests/ext/test_file.c
#   tests/ext/test_other_module.c
#   tests/test_top_level.c
# ---------------------------------------------------------------------------
file(GLOB_RECURSE c_test_files
    "${CMAKE_CURRENT_SOURCE_DIR}/test_*.c"
)

foreach(test_file ${c_test_files})
    add_c_test_executable("${test_file}")
endforeach()

# ---------------------------------------------------------------------------
# pytest — registered as a single CTest entry
# ---------------------------------------------------------------------------
add_test(
    NAME    python_tests
    COMMAND ${Python_EXECUTABLE} -m pytest "${CMAKE_SOURCE_DIR}/tests/" -v --tb=short
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)
