cmake_minimum_required(VERSION 3.16)

project(OpticalFlow2D LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_EXAMPLES "Build standalone optical flow executable" ON)
option(BUILD_TESTS "Build GoogleTest tests" OFF)
option(BUILD_MATLAB_MEX "Build Matlab MEX wrapper" ON)
option(BUILD_OCTAVE_MEX "Build Octave MEX wrapper" ON)


# ============================================================
# Core library - create static library to link against
# ============================================================

add_library(ImageRegistration
    src/ImageRegistration.cpp
    src/IterativeSolver.cpp
)

target_include_directories(ImageRegistration
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)


# ============================================================
# Standalone executable
# ============================================================

if(BUILD_EXAMPLES)

    if(EXISTS "${PROJECT_SOURCE_DIR}/include/json.hpp" AND
       EXISTS "${PROJECT_SOURCE_DIR}/include/CImg.h")

        add_executable(opticalflow2d_middlebury
            src/opticalflow2d_middlebury.cpp
        )

        target_link_libraries(opticalflow2d_middlebury
            PRIVATE ImageRegistration
        )

        target_include_directories(opticalflow2d_middlebury
            PRIVATE
                ${PROJECT_SOURCE_DIR}/include
        )

        add_custom_command(TARGET opticalflow2d_middlebury POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:opticalflow2d_middlebury>"
                "${PROJECT_SOURCE_DIR}/example/opticalflow2d_middlebury.exe"
        )

        message(STATUS "Standalone executable: ENABLED")

    else()

        message(STATUS "CImg.h and/or json.hpp not found in include directory -- skipping standalone executable")

    endif()

endif()


# ============================================================
# GoogleTest
# ============================================================

if(BUILD_TESTS)

    find_package(GTest QUIET)

    if(GTest_FOUND)

        enable_testing()

        add_executable(testOpticalFlow2d
            tests/mainTest.cpp

            tests/unit/testCoord2d.cpp
            tests/unit/testField.cpp
            tests/unit/testImage.cpp
            tests/unit/testMotion.cpp

            tests/integration/testConv2d.cpp
            tests/integration/testInterp2d.cpp
            tests/integration/testGradients.cpp
        )

        target_include_directories(testOpticalFlow2d
            PRIVATE
                ${PROJECT_SOURCE_DIR}/include
        )

        target_link_libraries(testOpticalFlow2d
            PRIVATE
                ImageRegistration
                GTest::gtest
        )

        add_test(
            NAME OpticalFlow2DTests
            COMMAND testOpticalFlow2d
        )

        message(STATUS "GoogleTest: ENABLED")

    else()

        message(STATUS "GoogleTest not found -- skipping tests")

    endif()

endif()


# ============================================================
# Octave MEX
# ============================================================
if(BUILD_OCTAVE_MEX)

    find_program(MKOCTFILE_EXECUTABLE
        NAMES mkoctfile
    )

    if(MKOCTFILE_EXECUTABLE)

        message(STATUS "mkoctfile found: ${MKOCTFILE_EXECUTABLE}")

        execute_process(
            COMMAND ${MKOCTFILE_EXECUTABLE} -p OCTINCLUDEDIR
            OUTPUT_VARIABLE OCTAVE_INCLUDE_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )

        if(EXISTS "${OCTAVE_INCLUDE_DIR}/mex.h")

            message(STATUS "Octave include directory: ${OCTAVE_INCLUDE_DIR}")
            message(STATUS "Octave MEX: ENABLED")

            add_custom_target(OpticalFlow2d ALL

                COMMAND
                    ${MKOCTFILE_EXECUTABLE}
                    --mex
                    -O3
                    -I${PROJECT_SOURCE_DIR}/include
                    -o
                    ${PROJECT_SOURCE_DIR}/mex/OpticalFlow2d.mex
                    ${PROJECT_SOURCE_DIR}/src/WrapperOpticalFlow2d.cpp
                    $<TARGET_FILE:ImageRegistration>

                DEPENDS
                    ImageRegistration

                WORKING_DIRECTORY
                    ${PROJECT_SOURCE_DIR}
            )

        else()

            message(STATUS
                "mex.h not found in ${OCTAVE_INCLUDE_DIR} -- skipping MEX"
            )

        endif()

    else()

        message(STATUS "mkoctfile not found -- skipping MEX")

    endif()

endif()

# ============================================================
# Matlab MEX
# ============================================================
if (BUILD_MATLAB_MEX)

    find_package(Matlab QUIET COMPONENTS MEX_COMPILER)

    if(Matlab_FOUND)
        message(STATUS "MATLAB found: ${Matlab_ROOT_DIR}")

        matlab_add_mex(
            NAME OpticalFlow2d
            SRC
                src/WrapperOpticalFlow2d.cpp
            LINK_TO
                ImageRegistration
        )

        target_include_directories(OpticalFlow2d
            PRIVATE
                ${PROJECT_SOURCE_DIR}/include
        )

        add_custom_command(TARGET OpticalFlow2d POST_BUILD
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                "$<TARGET_FILE:OpticalFlow2d>"
                "${PROJECT_SOURCE_DIR}/mex/$<TARGET_FILE_NAME:OpticalFlow2d>"
        )

    else()

        message(STATUS "MATLAB/MEX compiler not found -- skipping MEX")

    endif()


endif()