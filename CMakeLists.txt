cmake_minimum_required(VERSION 3.16)

project(OpticalFlow2D LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_EXAMPLES "Build standalone optical flow executable" ON)
option(BUILD_TESTS "Build GoogleTest tests" ON)
option(BUILD_MEX "Build Octave MEX wrapper" ON)


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
            example/opticalflow2d_middlebury.cpp
        )

        target_link_libraries(opticalflow2d_middlebury
            PRIVATE ImageRegistration
        )

        target_include_directories(opticalflow2d_middlebury
            PRIVATE
                ${PROJECT_SOURCE_DIR}/include
        )

        set_target_properties(opticalflow2d_middlebury
            PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/example"
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
if(BUILD_MEX)

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

            add_custom_target(OpticalFlow2d_mex ALL

                COMMAND
                    ${MKOCTFILE_EXECUTABLE}
                    --mex
                    -O3
                    -I${PROJECT_SOURCE_DIR}/include
                    -o
                    ${PROJECT_SOURCE_DIR}/mex/OpticalFlow2d.mex
                    -lImageRegistration
                    -L${PROJECT_SOURCE_DIR}/build

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