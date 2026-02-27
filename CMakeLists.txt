cmake_minimum_required(VERSION 3.22)
project(algos C)

find_package(Python 3.13.7 REQUIRED COMPONENTS Interpreter Development.Module NumPy)

# Find Cython
find_program(CYTHON_EXECUTABLE NAMES cython cython3 REQUIRED)

# Cython compiler directives
set(CYTHON_FLAGS
    --3str
    -X boundscheck=False
    -X wraparound=False
    -X initializedcheck=False
    -X nonecheck=False
    -X cdivision=True
    -X profile=False
)

# OpenMP detection
find_package(OpenMP)

# Helper function to register a .pyx extension
function(add_cython_extension module_name pyx_path)
    # Derive a safe target name from the module name (dots -> underscores)
    string(REPLACE "." "_" target_name ${module_name})

    # Output .c file sits next to the .pyx file
    get_filename_component(pyx_dir ${pyx_path} DIRECTORY)
    get_filename_component(pyx_stem ${pyx_path} NAME_WE)
    set(c_file "${CMAKE_SOURCE_DIR}/${pyx_dir}/${pyx_stem}.c")

    # Cython transpilation step: .pyx -> .c
    add_custom_command(
        OUTPUT ${c_file}
        COMMAND ${CYTHON_EXECUTABLE} ${CYTHON_FLAGS}
                -I ${CMAKE_SOURCE_DIR}
                ${CMAKE_SOURCE_DIR}/${pyx_path}
                -o ${c_file}
        DEPENDS ${CMAKE_SOURCE_DIR}/${pyx_path}
        COMMENT "Cythonizing ${pyx_path}"
    )

    # Compile the .c file into a Python extension
    Python_add_library(${target_name} MODULE ${c_file} WITH_SOABI)

    set_target_properties(${target_name} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/${pyx_dir}
        OUTPUT_NAME ${pyx_stem}
    )

    target_include_directories(${target_name} PRIVATE
        ${Python_NumPy_INCLUDE_DIRS}
    )

    target_compile_definitions(${target_name} PRIVATE
        NPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION
    )

    # Platform-specific compile flags
    target_compile_options(${target_name} PRIVATE
        $<$<NOT:$<C_COMPILER_ID:MSVC>>:-O3 -ffast-math>
        $<$<C_COMPILER_ID:MSVC>:/O2>
    )

    # OpenMP
    if(OpenMP_C_FOUND)
        target_link_libraries(${target_name} PRIVATE OpenMP::OpenMP_C)
    endif()

    # Install into the correct package subdirectory
    get_filename_component(install_dir ${pyx_path} DIRECTORY)
    install(TARGETS ${target_name} DESTINATION ${install_dir})
endfunction()
