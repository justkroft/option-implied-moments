cmake_minimum_required(VERSION 3.22)

option(TEST_COVERAGE "Build C tests with gcov/llvm-cov instrumentation" ON)

function(add_c_test_executable test_file)
    file(RELATIVE_PATH rel "${CMAKE_CURRENT_SOURCE_DIR}" "${test_file}")
    get_filename_component(rel_dir  "${rel}" DIRECTORY)
    get_filename_component(stem     "${rel}" NAME_WE)

    # Target names must not contain path separators (nested test dirs)
    if(rel_dir)
        string(REPLACE "/" "_" rel_dir_flat "${rel_dir}")
        set(target_name "${rel_dir_flat}_${stem}")
    else()
        set(target_name "${stem}")
    endif()

    # Strip the "test_" prefix from the stem for readability, then
    # combine with the subdirectory
    string(REGEX REPLACE "^test_" "" bare_stem "${stem}")
    if(rel_dir)
        set(ctest_name "${rel_dir}/${bare_stem}")
    else()
        set(ctest_name "${bare_stem}")
    endif()

    # Add test.c to every test executable
    set(TEST_HARNESS "${CMAKE_CURRENT_SOURCE_DIR}/test.c")

    # derive path to the source under test
    if(rel_dir)
        set(src_file "${CMAKE_SOURCE_DIR}/src/option_implied_moments/${rel_dir}/${bare_stem}.c")
    else()
        set(src_file "${CMAKE_SOURCE_DIR}/src/option_implied_moments/${bare_stem}.c")
    endif()

    # build executable
    if(EXISTS "${src_file}")
        get_filename_component(src_dir "${src_file}" DIRECTORY)
        add_executable("${target_name}" "${test_file}" "${src_file}" "${TEST_HARNESS}")
    else()
        set(src_dir "")
        add_executable("${target_name}" "${test_file}" "${TEST_HARNESS}")
    endif()

    target_include_directories("${target_name}" PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"   # test.h
        $<$<BOOL:${src_dir}>:${src_dir}>  # header next to the .c file, if it exists
    )

    if(MSVC)
        target_compile_options("${target_name}" PRIVATE /O2 /w)
    else()
        target_compile_options("${target_name}" PRIVATE -O2 -w)
    endif()

    if(NOT WIN32 AND NOT APPLE)
        target_link_libraries("${target_name}" PRIVATE m)
    endif()

    # Coverage flags (GCC / Clang only)
    if(TEST_COVERAGE AND CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options("${target_name}" PRIVATE --coverage)
        target_link_options("${target_name}"    PRIVATE --coverage)
    endif()

    set_target_properties("${target_name}" PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/tests/${rel_dir}"
    )

    add_test(
        NAME    "${ctest_name}"
        COMMAND "${target_name}"
    )

endfunction()

# Auto-discover all C test files under tests/
# CONFIGURE_DEPENDS re-runs the glob on build so new test files are picked up.
file(GLOB_RECURSE c_test_files CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/test_*.c"
)

foreach(test_file ${c_test_files})
    add_c_test_executable("${test_file}")
endforeach()

if(BUILD_EXTENSIONS)
    add_test(
        NAME    python_tests
        COMMAND ${Python_EXECUTABLE} -m pytest "${CMAKE_SOURCE_DIR}/tests/" -v --tb=short
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    )
endif()
