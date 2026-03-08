cmake_minimum_required(VERSION 3.22)
project(option-implied-moments C)

# ---------------------------------------------------------------------------
# Python + NumPy
# ---------------------------------------------------------------------------
find_package(Python 3.13 REQUIRED COMPONENTS Interpreter Development.Module NumPy)

# ---------------------------------------------------------------------------
# Cython executable
# ---------------------------------------------------------------------------
find_program(CYTHON_EXECUTABLE NAMES cython cython3 REQUIRED)

# Compiler directives for Cython
set(CYTHON_FLAGS
    --3str
    -X boundscheck=False
    -X wraparound=False
    -X initializedcheck=False
    -X nonecheck=False
    -X cdivision=True
    -X profile=False
)

# ---------------------------------------------------------------------------
# OpenMP
# ---------------------------------------------------------------------------
if(APPLE)
    find_program(BREW brew)
    if(BREW)
        execute_process(
            COMMAND ${BREW} --prefix libomp
            OUTPUT_VARIABLE LIBOMP_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if(EXISTS "${LIBOMP_PREFIX}")
            set(OpenMP_C_FLAGS "-Xclang -fopenmp")
            set(OpenMP_C_LIB_NAMES "omp")
            set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib")
            include_directories("${LIBOMP_PREFIX}/include")
            message(STATUS "macOS: found libomp at ${LIBOMP_PREFIX}")
        endif()
    endif()
endif()

find_package(OpenMP)

if(OpenMP_C_FOUND)
    message(STATUS "OpenMP found — parallel group iteration enabled")
else()
    message(WARNING
        "OpenMP not found. trapezoid_rnm will run single-threaded. "
        "On macOS install libomp via Homebrew: brew install libomp"
    )
endif()

# ---------------------------------------------------------------------------
# Helper: add_cython_extension(module_name  pyx_path  extra_c_sources)
#
#   module_name    : dotted Python module name, e.g. "src.ext.trapezoid_rnm"
#   pyx_path       : path to the .pyx file relative to CMAKE_SOURCE_DIR
#   extra_c_sources: list of additional .c files to compile into the same
#                    shared object (can be empty "")
# ---------------------------------------------------------------------------
function(add_cython_extension module_name pyx_path extra_c_sources)

    string(REPLACE "." "_" target_name ${module_name})

    get_filename_component(pyx_dir  ${pyx_path} DIRECTORY)
    get_filename_component(pyx_stem ${pyx_path} NAME_WE)

    # Cython writes the generated .c next to the .pyx file
    set(c_file "${CMAKE_SOURCE_DIR}/${pyx_dir}/${pyx_stem}.c")

    # ------------------------------------------------------------------
    # Collect all .pyx dependencies so re-cythonisation is triggered
    # when any .pxd file changes.
    # ------------------------------------------------------------------
    file(GLOB_RECURSE pxd_deps "${CMAKE_SOURCE_DIR}/${pyx_dir}/*.pxd")

    # ------------------------------------------------------------------
    # Cython transpilation:  .pyx  ->  .c
    # ------------------------------------------------------------------
    add_custom_command(
        OUTPUT  ${c_file}
        COMMAND ${CYTHON_EXECUTABLE}
                ${CYTHON_FLAGS}
                --include-dir ${CMAKE_SOURCE_DIR}
                # Needed for cython.parallel (prange)
                --include-dir ${CMAKE_SOURCE_DIR}/${pyx_dir}
                ${CMAKE_SOURCE_DIR}/${pyx_path}
                -o ${c_file}
        DEPENDS
            ${CMAKE_SOURCE_DIR}/${pyx_path}
            ${pxd_deps}
        COMMENT "Cythonizing ${pyx_path} → ${pyx_stem}.c"
        VERBATIM
    )

    # ------------------------------------------------------------------
    # Build the Python extension from:
    #   * the Cython-generated .c
    #   * any extra hand-written .c sources (e.g. trapezoid_core.c)
    # ------------------------------------------------------------------
    # Resolve extra_c_sources to absolute paths
    set(abs_extra_sources "")
    foreach(src ${extra_c_sources})
        if(IS_ABSOLUTE ${src})
            list(APPEND abs_extra_sources ${src})
        else()
            list(APPEND abs_extra_sources "${CMAKE_SOURCE_DIR}/${src}")
        endif()
    endforeach()

    Python_add_library(${target_name} MODULE
        ${c_file}
        ${abs_extra_sources}
        WITH_SOABI
    )

    # Output the .so next to the .pyx so Python can import it in-place
    set_target_properties(${target_name} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/${pyx_dir}"
        OUTPUT_NAME              "${pyx_stem}"
    )

    # ------------------------------------------------------------------
    # Include paths
    # ------------------------------------------------------------------
    target_include_directories(${target_name} PRIVATE
        ${Python_NumPy_INCLUDE_DIRS}
        # So the Cython .c can find trapezoid_core.h
        "${CMAKE_SOURCE_DIR}/${pyx_dir}"
    )

    # ------------------------------------------------------------------
    # Preprocessor definitions
    # ------------------------------------------------------------------
    target_compile_definitions(${target_name} PRIVATE
        NPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION
    )

    # ------------------------------------------------------------------
    # Compiler flags
    #
    #   -O3 -ffast-math   : maximum scalar throughput
    #   -fno-math-errno   : skip errno updates in libm (safe for our use)
    #   /fp:fast          : MSVC equivalent
    #
    # Note: -ffast-math implies -ffinite-math-only which removes NaN
    # checks.  Our NaN propagation relies on IEEE bit patterns, not on
    # floating-point comparisons, so this is safe here.
    # ------------------------------------------------------------------
    target_compile_options(${target_name} PRIVATE
        $<$<NOT:$<C_COMPILER_ID:MSVC>>:-O3 -ffast-math -fno-math-errno>
        $<$<C_COMPILER_ID:MSVC>:/O2 /fp:fast>
    )

    # ------------------------------------------------------------------
    # OpenMP
    # ------------------------------------------------------------------
    if(OpenMP_C_FOUND)
        target_link_libraries(${target_name} PRIVATE OpenMP::OpenMP_C)
        target_compile_options(${target_name} PRIVATE
            $<$<NOT:$<C_COMPILER_ID:MSVC>>:${OpenMP_C_FLAGS}>
        )
    endif()

    # ------------------------------------------------------------------
    # Link math library (required on Linux, no-op elsewhere)
    # ------------------------------------------------------------------
    target_link_libraries(${target_name} PRIVATE
        $<$<NOT:$<C_COMPILER_ID:MSVC>>:m>
    )

    # ------------------------------------------------------------------
    # Install: mirror the source tree under the install prefix
    # ------------------------------------------------------------------
    install(TARGETS ${target_name}
            LIBRARY DESTINATION "${pyx_dir}"
    )

endfunction()

# ---------------------------------------------------------------------------
# Register extensions
#
# add_cython_extension(
#     <module_name>
#     <path/to/module.pyx>
#     "<path/to/extra1.c;path/to/extra2.c>"
# )
# ---------------------------------------------------------------------------
add_cython_extension(
    "src.ext.trapezoid_rnm"
    "src/ext/trapezoid_rnm.pyx"
    "src/ext/trapezoid_core.c"
)

option(TESTING "Build tests for C extensions" ON)
if (TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()
