cmake_minimum_required(VERSION 3.22)
project(option-implied-moments C)

# Build the Cython extension modules. Turn OFF for a pure-C test build
# (`make test-c`): no Python, Cython, NumPy or OpenMP needed, and nothing
# is written into src/.
option(BUILD_EXTENSIONS "Build the Cython extension modules" ON)

option(TESTING "Build tests for C extensions" OFF)

if(BUILD_EXTENSIONS)
    find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module NumPy)

    # Use the Cython of the *active* Python (the isolated build env under
    # scikit-build-core) instead of whichever `cython` happens to be on PATH.
    execute_process(
        COMMAND ${Python_EXECUTABLE} -m cython --version
        RESULT_VARIABLE CYTHON_CHECK_RESULT
        OUTPUT_QUIET ERROR_QUIET
    )
    if(NOT CYTHON_CHECK_RESULT EQUAL 0)
        message(FATAL_ERROR
            "Cython not importable by ${Python_EXECUTABLE}. "
            "Install it with: pip install 'cython>=3.1.3'"
        )
    endif()

    set(CYTHON_FLAGS
        --3str
        -X boundscheck=False
        -X wraparound=False
        -X initializedcheck=False
        -X nonecheck=False
        -X cdivision=True
        -X profile=False
    )

    # AppleClang needs Homebrew libomp; keep the compiler selected by project()
    # and let FindOpenMP use the -Xpreprocessor recipe.
    if(APPLE AND CMAKE_C_COMPILER_ID STREQUAL "AppleClang")
        set(_libomp_hints "")
        if(DEFINED ENV{LIBOMP_PREFIX})
            list(APPEND _libomp_hints "$ENV{LIBOMP_PREFIX}")
        endif()
        find_program(BREW brew)
        if(BREW)
            execute_process(
                COMMAND ${BREW} --prefix libomp
                OUTPUT_VARIABLE _brew_libomp
                RESULT_VARIABLE _brew_rc
                ERROR_QUIET
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )
            if(_brew_rc EQUAL 0 AND _brew_libomp)
                list(APPEND _libomp_hints "${_brew_libomp}")
            endif()
        endif()
        # Apple Silicon and Intel Homebrew defaults
        list(APPEND _libomp_hints /opt/homebrew/opt/libomp /usr/local/opt/libomp)

        find_path(LIBOMP_INCLUDE_DIR omp.h
            HINTS ${_libomp_hints} PATH_SUFFIXES include NO_DEFAULT_PATH)
        find_library(LIBOMP_LIBRARY omp
            HINTS ${_libomp_hints} PATH_SUFFIXES lib NO_DEFAULT_PATH)

        if(LIBOMP_INCLUDE_DIR AND LIBOMP_LIBRARY)
            set(OpenMP_C_FLAGS
                "-Xpreprocessor -fopenmp -I${LIBOMP_INCLUDE_DIR}"
                CACHE STRING "" FORCE)
            set(OpenMP_C_LIB_NAMES "omp" CACHE STRING "" FORCE)
            set(OpenMP_omp_LIBRARY "${LIBOMP_LIBRARY}" CACHE FILEPATH "" FORCE)
            message(STATUS "macOS: using libomp from ${LIBOMP_LIBRARY}")
        endif()
    endif()

    find_package(OpenMP COMPONENTS C)

    if(OpenMP_C_FOUND)
        message(STATUS "OpenMP found - parallel group iteration enabled")
    else()
        message(WARNING
            "OpenMP not found. trapezoid_rnm will run single-threaded. "
            "On macOS install libomp via Homebrew: brew install libomp"
        )
    endif()
endif()

# Function to add a Cython extension module
#   module_name     : Python module name, e.g. "src.ext.trapezoid_rnm"
#   pyx_path        : path to the .pyx file relative to CMAKE_SOURCE_DIR
#   extra_c_sources : list of additional .c files to compile into the same
#                     shared object (can be empty "")
function(add_cython_extension module_name pyx_path extra_c_sources)

    string(REPLACE "." "_" target_name ${module_name})

    get_filename_component(pyx_dir  ${pyx_path} DIRECTORY)
    get_filename_component(pyx_stem ${pyx_path} NAME_WE)

    # Generate C under the build tree to avoid reusing stale sources in src/.
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${pyx_dir}")
    set(c_file "${CMAKE_CURRENT_BINARY_DIR}/${pyx_dir}/${pyx_stem}.c")

    # Collect all .pxd dependencies so re-cythonisation is triggered
    # when any .pxd file changes.
    file(GLOB_RECURSE pxd_deps "${CMAKE_SOURCE_DIR}/${pyx_dir}/*.pxd")

    add_custom_command(
        OUTPUT  ${c_file}
        COMMAND ${Python_EXECUTABLE} -m cython
                ${CYTHON_FLAGS}
                --include-dir ${CMAKE_SOURCE_DIR}
                # Needed for cython.parallel (prange)
                --include-dir ${CMAKE_SOURCE_DIR}/${pyx_dir}
                ${CMAKE_SOURCE_DIR}/${pyx_path}
                -o ${c_file}
        DEPENDS
            ${CMAKE_SOURCE_DIR}/${pyx_path}
            ${pxd_deps}
        COMMENT "Cythonizing ${pyx_path} -> ${pyx_stem}.c"
        VERBATIM
    )

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
        LIBRARY_OUTPUT_DIRECTORY         "${CMAKE_SOURCE_DIR}/${pyx_dir}"
        LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/${pyx_dir}"
        LIBRARY_OUTPUT_DIRECTORY_DEBUG   "${CMAKE_SOURCE_DIR}/${pyx_dir}"
        OUTPUT_NAME                      "${pyx_stem}"
    )

    target_include_directories(${target_name} PRIVATE
        ${Python_NumPy_INCLUDE_DIRS}
        "${CMAKE_SOURCE_DIR}/${pyx_dir}"
    )

    target_compile_definitions(${target_name} PRIVATE
        NPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION
    )

    target_compile_options(${target_name} PRIVATE
        $<$<NOT:$<C_COMPILER_ID:MSVC>>:-O3 -fno-math-errno -fno-trapping-math -fno-finite-math-only -ffp-contract=on>
        $<$<AND:$<C_COMPILER_ID:MSVC>,$<CONFIG:Release>>:/O2 /fp:precise>
        $<$<AND:$<C_COMPILER_ID:MSVC>,$<NOT:$<CONFIG:Release>>>:/fp:precise>
    )

    if(OpenMP_C_FOUND)
        target_link_libraries(${target_name} PRIVATE OpenMP::OpenMP_C)
    endif()

    if(NOT WIN32 AND NOT APPLE)
        target_link_libraries(${target_name} PRIVATE m)
    endif()

    # mirror the source tree under the install prefix
    string(REGEX REPLACE "^src/" "" install_dir "${pyx_dir}")
    install(TARGETS ${target_name}
        LIBRARY DESTINATION "${install_dir}"
    )

endfunction()

# Register extensions
#
# add_cython_extension(
#     <module_name>
#     <path/to/module.pyx>
#     "<path/to/extra1.c;path/to/extra2.c>"
# )
if(BUILD_EXTENSIONS)
    add_cython_extension(
        "option_implied_moments.ext.trapezoid_rnm"
        "src/option_implied_moments/ext/trapezoid_rnm.pyx"
        "src/option_implied_moments/ext/trapezoid_core.c"
    )
    add_cython_extension(
        "option_implied_moments.ext.omp_utils"
        "src/option_implied_moments/ext/omp_utils.pyx"
        ""
    )
endif()

if(TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()
