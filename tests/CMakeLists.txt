# Unit tests for TORTOISEV4 motion/eddy/s2v pipeline
# Build with: cmake -DBUILD_TESTS=ON ..

# Re-find dependencies in this scope (parent's find_package doesn't fully propagate to subdirectories)
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})
find_package(Boost COMPONENTS iostreams filesystem system REQUIRED)
find_package(OpenMP REQUIRED)

# Common include directories
set(TEST_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/../src/main
    ${CMAKE_SOURCE_DIR}/../src/utilities
    ${CMAKE_SOURCE_DIR}/../src/external_src
    ${CMAKE_SOURCE_DIR}/../tests
    ${CMAKE_SOURCE_DIR}/../src/tools/DRBUDDI
)

# Common sources needed by multiple tests (redefine SOURCES locally for tests)
set(COMMON_SOURCES
    ${CMAKE_SOURCE_DIR}/../src/main/defines.cxx
    ${CMAKE_SOURCE_DIR}/../src/main/TORTOISE_global.cxx
    ${CMAKE_SOURCE_DIR}/../src/main/antsCommandLineParser.cxx
    ${CMAKE_SOURCE_DIR}/../src/main/antsCommandLineOption.cxx
    ${CMAKE_SOURCE_DIR}/../src/utilities/TORTOISE_Utilities.cxx
    ${CMAKE_SOURCE_DIR}/../src/utilities/extract_3Dvolume_from_4D.cxx
    ${CMAKE_SOURCE_DIR}/../src/utilities/read_3Dvolume_from_4D.cxx
    ${CMAKE_SOURCE_DIR}/../src/utilities/read_bmatrix_file.cxx
    ${CMAKE_SOURCE_DIR}/../src/utilities/math_utilities.cxx
    ${CMAKE_SOURCE_DIR}/../src/external_src/cmpfit-1.3a/mpfit.c
)


# ============================================================
# Tier 1: Math utilities test (lightweight ITK dependency)
# ============================================================
add_executable(test_math_utilities test_math_utilities.cxx
    ${CMAKE_SOURCE_DIR}/../src/utilities/math_utilities.cxx
    ${CMAKE_SOURCE_DIR}/../src/main/defines.cxx)
target_include_directories(test_math_utilities PRIVATE ${TEST_INCLUDE_DIRS})
target_link_libraries(test_math_utilities ${ITK_LIBRARIES} ${Boost_LIBRARIES})
add_test(NAME MathUtilities COMMAND test_math_utilities)


# ============================================================
# Tier 2: OkanQuadraticTransform correctness (ITK only)
# ============================================================
add_executable(test_okan_transform test_okan_transform.cxx)
target_include_directories(test_okan_transform PRIVATE ${TEST_INCLUDE_DIRS})
target_link_libraries(test_okan_transform ${ITK_LIBRARIES})
add_test(NAME OkanTransform COMMAND test_okan_transform)


# ============================================================
# Tier 2: Jacobian validation against finite differences (ITK only)
# Catches bugs on lines 538 and 624 of itkOkanQuadraticTransform.hxx
# ============================================================
add_executable(test_okan_jacobian test_okan_jacobian.cxx)
target_include_directories(test_okan_jacobian PRIVATE ${TEST_INCLUDE_DIRS})
target_link_libraries(test_okan_jacobian ${ITK_LIBRARIES})
add_test(NAME OkanJacobian COMMAND test_okan_jacobian)


# ============================================================
# Tier 3: Registration recovery (full pipeline test)
# ============================================================
add_executable(test_registration_recovery
    test_registration_recovery.cxx
    ${CMAKE_SOURCE_DIR}/../src/main/register_dwi_to_b0.cxx
    ${COMMON_SOURCES})
target_include_directories(test_registration_recovery PRIVATE ${TEST_INCLUDE_DIRS})
target_link_libraries(test_registration_recovery
    ${ITK_LIBRARIES} ${Boost_LIBRARIES} OpenMP::OpenMP_CXX)
add_test(NAME RegistrationRecovery COMMAND test_registration_recovery)


# ============================================================
# Tier 3: S2V forward transform (KD-tree interpolation)
# ============================================================
add_executable(test_s2v_transform
    test_s2v_transform.cxx
    ${COMMON_SOURCES})
target_include_directories(test_s2v_transform PRIVATE ${TEST_INCLUDE_DIRS})
target_link_libraries(test_s2v_transform
    ${ITK_LIBRARIES} ${Boost_LIBRARIES} OpenMP::OpenMP_CXX)
add_test(NAME S2VTransform COMMAND test_s2v_transform)


# ============================================================
# Tier 2: Outlier detection math (EM clustering, ComputeResidProb, MAD)
# ============================================================
add_executable(test_outlier_detection test_outlier_detection.cxx
    ${CMAKE_SOURCE_DIR}/../src/utilities/math_utilities.cxx
    ${CMAKE_SOURCE_DIR}/../src/main/defines.cxx)
target_include_directories(test_outlier_detection PRIVATE ${TEST_INCLUDE_DIRS})
target_link_libraries(test_outlier_detection ${ITK_LIBRARIES} ${Boost_LIBRARIES})
add_test(NAME OutlierDetection COMMAND test_outlier_detection)


# ============================================================
# Tier 1: Slice timing JSON -> slspec conversion
# ============================================================
add_executable(test_slice_timing test_slice_timing.cxx
    ${CMAKE_SOURCE_DIR}/../src/main/defines.cxx)
target_include_directories(test_slice_timing PRIVATE ${TEST_INCLUDE_DIRS})
target_link_libraries(test_slice_timing ${ITK_LIBRARIES} ${Boost_LIBRARIES})
add_test(NAME SliceTiming COMMAND test_slice_timing)


# ============================================================
# Tier 3: Displacement field round-trip (transform <-> field)
# ============================================================
add_executable(test_displacement_field
    test_displacement_field.cxx
    ${COMMON_SOURCES})
target_include_directories(test_displacement_field PRIVATE
    ${TEST_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/../src/tools/ConvertQuadraticTransformToDisplacementField)
target_link_libraries(test_displacement_field
    ${ITK_LIBRARIES} ${Boost_LIBRARIES} OpenMP::OpenMP_CXX)
add_test(NAME DisplacementField COMMAND test_displacement_field)


# ============================================================
# Tier 3: S2V registration end-to-end (VolumeToSliceRegistration)
# ============================================================
add_executable(test_s2v_registration
    test_s2v_registration.cxx
    ${COMMON_SOURCES})
target_include_directories(test_s2v_registration PRIVATE ${TEST_INCLUDE_DIRS})
target_link_libraries(test_s2v_registration
    ${ITK_LIBRARIES} ${Boost_LIBRARIES} OpenMP::OpenMP_CXX)
add_test(NAME S2VRegistration COMMAND test_s2v_registration)
