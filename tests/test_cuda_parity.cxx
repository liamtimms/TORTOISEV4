// CUDA parity tests for the S2V registration new parameters.
// When built with USECUDA=0, all tests pass trivially (skipped).
// When built with USECUDA=1, tests verify the CUDA path handles
// warm_start and smoothing_sigma correctly.
//
// NOTE: Known behavioral difference between CPU and CUDA paths:
// - CPU (register_dwi_to_slice.h): When smoothing_sigma > 0, uses
//   two-level registration (smoothed pass at level 0, then unsmoothed
//   pass at level 1).
// - CUDA (register_dwi_to_slice_cuda.h): When smoothing_sigma > 0,
//   applies Gaussian smoothing to the fixed image once, then runs a
//   single optimization pass. No two-level hierarchy.
// This means results may differ slightly between CPU and CUDA builds
// when smoothing is active.

#include "test_macros.h"
#include "itkOkanQuadraticTransform.h"

#ifdef USECUDA
#include "register_dwi_to_slice_cuda.h"
#include "test_image_helpers.h"
#endif


// ============================================================
// Test: CUDA warm_start initialization matches CPU logic
// ============================================================
void test_cuda_warm_start_matches_cpu()
{
#ifdef USECUDA
    // Both CPU and CUDA paths use the same logic:
    //   if(!warm_start || s2v_transformations.size() != sz[2])
    //       reset to identity
    // Verify by pre-filling transforms and checking preservation

    int sx = 16, sy = 16, sz = 8;
    auto img = create_test_image(sx, sy, sz, 2.0, 2.0, 2.0, 500.0f);
    std::vector<float> lim_arr = {0.0f, 1000.0f, 0.0f, 1000.0f};

    vnl_matrix<int> slspec(sz, 1);
    for (int k = 0; k < sz; k++) slspec(k, 0) = k;

    using TransformType = itk::OkanQuadraticTransform<double, 3, 3>;

    // Pre-fill transforms with Tz = 1mm
    std::vector<TransformType::Pointer> transforms(sz);
    for (int k = 0; k < sz; k++) {
        transforms[k] = TransformType::New();
        transforms[k]->SetPhase("horizontal");
        transforms[k]->SetIdentity();
        auto p = transforms[k]->GetParameters();
        p[2] = 1.0;
        transforms[k]->SetParameters(p);
    }

    // warm_start=true should preserve
    VolumeToSliceRegistration_cuda(img, img, slspec, lim_arr,
                                    transforms, false, "horizontal", nullptr, 0, true, 0.0f);

    bool any_preserved = false;
    for (int k = 0; k < sz; k++) {
        if (std::fabs(transforms[k]->GetParameters()[2]) > 0.3)
            any_preserved = true;
    }
    ASSERT_TRUE(any_preserved);
#else
    // No CUDA — test passes trivially
    ASSERT_TRUE(true);
#endif
}


// ============================================================
// Test: CUDA smoothing path activates when sigma > 0
// ============================================================
void test_cuda_smoothing_applied()
{
#ifdef USECUDA
    int sx = 16, sy = 16, sz = 8;
    auto img = create_test_image(sx, sy, sz, 2.0, 2.0, 2.0, 500.0f);
    std::vector<float> lim_arr = {0.0f, 1000.0f, 0.0f, 1000.0f};

    vnl_matrix<int> slspec(sz, 1);
    for (int k = 0; k < sz; k++) slspec(k, 0) = k;

    using TransformType = itk::OkanQuadraticTransform<double, 3, 3>;
    std::vector<TransformType::Pointer> transforms;

    // Should complete without crash with smoothing_sigma > 0
    VolumeToSliceRegistration_cuda(img, img, slspec, lim_arr,
                                    transforms, false, "horizontal", nullptr, 0, false, 2.0f);

    ASSERT_EQ((int)transforms.size(), sz);

    // All params should be finite
    for (int k = 0; k < sz; k++) {
        auto p = transforms[k]->GetParameters();
        for (unsigned int pp = 0; pp < p.GetSize(); pp++)
            ASSERT_TRUE(std::isfinite(p[pp]));
    }
#else
    ASSERT_TRUE(true);
#endif
}


// ============================================================
// Test: CUDA function signature has warm_start and smoothing_sigma
// ============================================================
void test_cuda_signature_parity()
{
#ifdef USECUDA
    // Compile-time check: this test compiles only if the CUDA function
    // has the expected signature with warm_start and smoothing_sigma params.
    // If the signature doesn't match, this file won't compile.
    using TransformType = itk::OkanQuadraticTransform<double, 3, 3>;
    std::vector<TransformType::Pointer> dummy;
    vnl_matrix<int> dummy_slspec(1, 1);
    dummy_slspec(0, 0) = 0;
    std::vector<float> dummy_lim = {0, 1, 0, 1};

    // Verify the function accepts all 11 parameters including the new ones
    // (This is a compile-time check — we don't actually run registration)
    (void)static_cast<void(*)(ImageType3D::Pointer, ImageType3D::Pointer,
                               vnl_matrix<int>, std::vector<float>,
                               std::vector<TransformType::Pointer>&,
                               bool, std::string, ImageType3D::Pointer,
                               int, bool, float)>(
        &VolumeToSliceRegistration_cuda);
    ASSERT_TRUE(true);
#else
    // On CPU-only build, just verify the test infrastructure works
    ASSERT_TRUE(true);
#endif
}


int main()
{
    std::cout << "=== CUDA Parity Tests ===" << std::endl;

#ifndef USECUDA
    std::cout << "  (USECUDA not defined — CUDA tests skipped)" << std::endl;
#endif

    TEST(cuda_warm_start_matches_cpu);
    TEST(cuda_smoothing_applied);
    TEST(cuda_signature_parity);

    TEST_SUMMARY();
}
