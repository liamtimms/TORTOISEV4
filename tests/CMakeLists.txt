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
    ${CMAKE_SOURCE_DIR}/../src/tools/ResampleDWIs
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
    ${CMAKE_SOURCE_DIR}/../src/main/rigid_register_images.cxx
    ${CMAKE_SOURCE_DIR}/../src/tools/ResampleDWIs/resample_dwis.cxx
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
# Tier 0: Schedule parsing (zero external deps)
# ============================================================
add_executable(test_schedule_parsing test_schedule_parsing.cxx)
target_include_directories(test_schedule_parsing PRIVATE
    ${CMAKE_SOURCE_DIR}/../src/main)
add_test(NAME ScheduleParsing COMMAND test_schedule_parsing)


# ============================================================
# Tier 2: Temporal regularization math (ITK only)
# ============================================================
add_executable(test_temporal_regularize test_temporal_regularize.cxx)
target_include_directories(test_temporal_regularize PRIVATE
    ${TEST_INCLUDE_DIRS})
target_link_libraries(test_temporal_regularize ${ITK_LIBRARIES})
add_test(NAME TemporalRegularize COMMAND test_temporal_regularize)


# ============================================================
# Tier 0: Compile-only verification for modified headers
# ============================================================
add_executable(test_compile_headers test_compile_headers.cxx)
target_include_directories(test_compile_headers PRIVATE
    ${TEST_INCLUDE_DIRS})
target_link_libraries(test_compile_headers ${ITK_LIBRARIES})
add_test(NAME CompileHeaders COMMAND test_compile_headers)


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


# ============================================================
# Tier 1: CLI argument parsing for new S2V settings
# ============================================================
add_executable(test_cli_parsing test_cli_parsing.cxx
    ${CMAKE_SOURCE_DIR}/../src/main/TORTOISE_parser.cxx
    ${CMAKE_SOURCE_DIR}/../src/tools/DRBUDDI/DRBUDDI_parserBase.cxx
    ${CMAKE_SOURCE_DIR}/../src/main/antsCommandLineParser.cxx
    ${CMAKE_SOURCE_DIR}/../src/main/antsCommandLineOption.cxx
    ${CMAKE_SOURCE_DIR}/../src/main/defines.cxx)
target_include_directories(test_cli_parsing PRIVATE
    ${TEST_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/../src/tools/DRBUDDI)
target_link_libraries(test_cli_parsing ${ITK_LIBRARIES} ${Boost_LIBRARIES})
add_test(NAME CLIParsing COMMAND test_cli_parsing)


# ============================================================
# Tier 1: Convergence metric computation logic
# ============================================================
add_executable(test_convergence_logic test_convergence_logic.cxx)
target_include_directories(test_convergence_logic PRIVATE ${TEST_INCLUDE_DIRS})
target_link_libraries(test_convergence_logic ${ITK_LIBRARIES})
add_test(NAME ConvergenceLogic COMMAND test_convergence_logic)


# ============================================================
# Tier 3: S2V epoch loop integration (warm-start, smoothing, niter, convergence)
# ============================================================
add_executable(test_s2v_epoch_loop
    test_s2v_epoch_loop.cxx
    ${COMMON_SOURCES})
target_include_directories(test_s2v_epoch_loop PRIVATE ${TEST_INCLUDE_DIRS})
target_link_libraries(test_s2v_epoch_loop
    ${ITK_LIBRARIES} ${Boost_LIBRARIES} OpenMP::OpenMP_CXX)
add_test(NAME S2VEpochLoop COMMAND test_s2v_epoch_loop)


# ============================================================
# Tier 0: CUDA parity verification (skipped when USECUDA=0)
# ============================================================
add_executable(test_cuda_parity test_cuda_parity.cxx)
target_include_directories(test_cuda_parity PRIVATE ${TEST_INCLUDE_DIRS})
target_link_libraries(test_cuda_parity ${ITK_LIBRARIES})
add_test(NAME CUDAParity COMMAND test_cuda_parity)
