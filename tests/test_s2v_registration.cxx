// End-to-end tests for VolumeToSliceRegistration from register_dwi_to_slice.h:
// verifies identity recovery, per-slice shift recovery, multiband grouping,
// insufficient-voxel handling, and the do_eddy flag.

#include "test_macros.h"
#include "test_image_helpers.h"
#include "register_dwi_to_slice.h"
#include <cmath>

using TransformType = OkanQuadraticTransformType;


// --- Helper: create a structured test image with slice-dependent pattern ---
static ImageType3D::Pointer create_s2v_test_image(int sx, int sy, int sz,
                                                     double sp = 2.0)
{
    auto img = create_test_image(sx, sy, sz, sp, sp, sp, 0.0f);

    // Fill with a pattern that provides good MI contrast:
    // Gaussian blob + sinusoidal modulation + slice-dependent offset
    double cx = (sx - 1) * sp / 2.0;
    double cy = (sy - 1) * sp / 2.0;
    double cz = (sz - 1) * sp / 2.0;

    itk::ImageRegionIteratorWithIndex<ImageType3D> it(img, img->GetLargestPossibleRegion());
    for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
        ImageType3D::PointType pt;
        img->TransformIndexToPhysicalPoint(it.GetIndex(), pt);

        double dx = (pt[0] - cx) / (sx * sp * 0.3);
        double dy = (pt[1] - cy) / (sy * sp * 0.3);
        double dz = (pt[2] - cz) / (sz * sp * 0.3);
        double r2 = dx * dx + dy * dy + dz * dz;

        // Gaussian envelope with sinusoidal modulation
        float val = 800.0f * std::exp(-0.5 * r2) * (1.0f + 0.3f * std::sin(3.0 * std::sqrt(r2)));
        // Add slice-dependent constant for per-slice differentiation
        val += 200.0f + 50.0f * it.GetIndex()[2];

        it.Set(val);
    }
    return img;
}


// --- Helper: compute lim_arr for MI metric from image content ---
static std::vector<float> compute_lim_arr(ImageType3D::Pointer fixed,
                                            ImageType3D::Pointer moving)
{
    auto fr = compute_image_range(fixed);
    auto mr = compute_image_range(moving);
    return {fr.first, fr.second, mr.first, mr.second};
}


// --- Helper: create MB=1 slspec (identity) ---
static vnl_matrix<int> make_slspec_mb1(int nslices)
{
    vnl_matrix<int> slspec(nslices, 1);
    for (int k = 0; k < nslices; k++)
        slspec(k, 0) = k;
    return slspec;
}


// --- Helper: create MB=2 slspec (regular interleaving) ---
static vnl_matrix<int> make_slspec_mb2(int nslices)
{
    int MB = 2;
    int Nexc = nslices / MB;
    vnl_matrix<int> slspec(Nexc, MB);
    slspec.fill(0);
    for (int k = 0; k < nslices; k++) {
        int r = k % Nexc;
        int c = k / Nexc;
        slspec(r, c) = k;
    }
    return slspec;
}


// --- Helper: max rigid parameter magnitude across all transforms ---
static double max_rigid_param(const std::vector<TransformType::Pointer>& transforms)
{
    double max_val = 0;
    for (auto& t : transforms) {
        auto p = t->GetParameters();
        for (int i = 0; i < 6; i++) {
            double v = std::fabs(p[i]);
            if (v > max_val) max_val = v;
        }
    }
    return max_val;
}


// ============================================================
// Test: identical fixed/moving produces near-identity transforms
// ============================================================
void test_s2v_identity_no_motion()
{
    int sx = 16, sy = 16, sz = 8;
    auto img = create_s2v_test_image(sx, sy, sz);
    auto mask = create_block_mask(sx, sy, sz, 2.0, 2.0, 2.0, 1);
    auto lim_arr = compute_lim_arr(img, img);
    auto slspec = make_slspec_mb1(sz);

    std::vector<TransformType::Pointer> s2v_transforms;

    VolumeToSliceRegistration(img, img, slspec, lim_arr,
                              s2v_transforms, false, "horizontal", mask, 0);

    ASSERT_EQ((int)s2v_transforms.size(), sz);

    // All transforms should be near-identity.
    // The 2-slice window approach with doubled slice spacing makes the
    // through-plane (Z) direction less constrained, so allow wider tolerance there.
    for (int k = 0; k < sz; k++) {
        auto params = s2v_transforms[k]->GetParameters();
        // In-plane translation (0-1) should be near zero
        ASSERT_NEAR(params[0], 0.0, 0.5);
        ASSERT_NEAR(params[1], 0.0, 0.5);
        // Through-plane translation (2) has wider tolerance due to 2-slice windows
        ASSERT_NEAR(params[2], 0.0, 1.5);
        // Rotation parameters (3-5) should be near zero
        ASSERT_NEAR(params[3], 0.0, 0.03);
        ASSERT_NEAR(params[4], 0.0, 0.03);
        ASSERT_NEAR(params[5], 0.0, 0.03);
    }
}


// ============================================================
// Test: recovers per-slice shift in phase direction
// ============================================================
void test_s2v_recovers_per_slice_shift()
{
    int sx = 24, sy = 24, sz = 12;
    auto fixed = create_s2v_test_image(sx, sy, sz);
    auto mask = create_block_mask(sx, sy, sz, 2.0, 2.0, 2.0, 1);

    // Create moving image with slices 5-7 shifted by +1 voxel (2mm) in X (phase)
    using DupType = itk::ImageDuplicator<ImageType3D>;
    auto dup = DupType::New();
    dup->SetInputImage(fixed);
    dup->Update();
    auto moving = dup->GetOutput();

    // Shift slices 5-7 by copying pixel data with offset
    for (int k = 5; k <= 7; k++) {
        ImageType3D::IndexType idx;
        idx[2] = k;
        for (int j = 0; j < sy; j++) {
            idx[1] = j;
            for (int i = 0; i < sx - 1; i++) {
                idx[0] = i;
                ImageType3D::IndexType src_idx = idx;
                src_idx[0] = i + 1;
                moving->SetPixel(idx, fixed->GetPixel(src_idx));
            }
            // Last column: extrapolate with neighbor value
            idx[0] = sx - 1;
            ImageType3D::IndexType prev_idx = idx;
            prev_idx[0] = sx - 2;
            moving->SetPixel(idx, fixed->GetPixel(prev_idx));
        }
    }

    auto lim_arr = compute_lim_arr(fixed, moving);
    auto slspec = make_slspec_mb1(sz);

    std::vector<TransformType::Pointer> s2v_transforms;

    VolumeToSliceRegistration(fixed, moving, slspec, lim_arr,
                              s2v_transforms, false, "horizontal", mask, 0);

    ASSERT_EQ((int)s2v_transforms.size(), sz);

    // Shifted slices should have detectable translation in the X direction
    // The exact magnitude depends on the registration convergence
    // For undisplaced slices far from the shift region, transforms should be small
    for (int k = 0; k <= 2; k++) {
        auto p = s2v_transforms[k]->GetParameters();
        // These slices should have small motion
        ASSERT_TRUE(std::fabs(p[0]) < 2.0);
    }

    // For shifted slices (5-7), at least one should show detectable shift
    bool any_shift_detected = false;
    for (int k = 5; k <= 7; k++) {
        auto p = s2v_transforms[k]->GetParameters();
        if (std::fabs(p[0]) > 0.3)
            any_shift_detected = true;
    }
    ASSERT_TRUE(any_shift_detected);
}


// ============================================================
// Test: multiband grouping — same excitation gets same transform
// ============================================================
void test_s2v_multiband_grouping()
{
    int sx = 16, sy = 16, sz = 8;
    auto img = create_s2v_test_image(sx, sy, sz);
    auto mask = create_block_mask(sx, sy, sz, 2.0, 2.0, 2.0, 1);
    auto lim_arr = compute_lim_arr(img, img);
    auto slspec = make_slspec_mb2(sz);

    // slspec for 8 slices, MB=2:
    // Exc 0: slices 0, 4
    // Exc 1: slices 1, 5
    // Exc 2: slices 2, 6
    // Exc 3: slices 3, 7

    std::vector<TransformType::Pointer> s2v_transforms;

    VolumeToSliceRegistration(img, img, slspec, lim_arr,
                              s2v_transforms, false, "horizontal", mask, 0);

    ASSERT_EQ((int)s2v_transforms.size(), sz);

    // All MB slices in the same excitation group should get the same transform
    int Nexc = slspec.rows();
    int MB = slspec.cols();
    for (int e = 0; e < Nexc; e++) {
        auto params_ref = s2v_transforms[slspec(e, 0)]->GetParameters();
        for (int m = 1; m < MB; m++) {
            auto params_mb = s2v_transforms[slspec(e, m)]->GetParameters();
            for (int pp = 0; pp < 6; pp++) {
                ASSERT_NEAR(params_ref[pp], params_mb[pp], 1e-10);
            }
        }
    }
}


// ============================================================
// Test: insufficient voxels — no crash, returns identity-like
// ============================================================
void test_s2v_insufficient_voxels()
{
    int sx = 16, sy = 16, sz = 8;
    // Create nearly-zero image
    auto img = create_test_image(sx, sy, sz, 2.0, 2.0, 2.0, 0.01f);
    // Create mask with very few nonzero voxels (less than 10% of slice)
    auto mask = create_test_image(sx, sy, sz, 2.0, 2.0, 2.0, 0.0f);
    // Only set a tiny corner
    ImageType3D::IndexType idx;
    for (int k = 0; k < sz; k++) {
        idx[2] = k;
        for (int j = 0; j < 2; j++) {
            idx[1] = j;
            for (int i = 0; i < 2; i++) {
                idx[0] = i;
                mask->SetPixel(idx, 1.0f);
            }
        }
    }

    std::vector<float> lim_arr = {0.001f, 1.0f, 0.001f, 1.0f};
    auto slspec = make_slspec_mb1(sz);

    std::vector<TransformType::Pointer> s2v_transforms;

    // Should not crash
    VolumeToSliceRegistration(img, img, slspec, lim_arr,
                              s2v_transforms, false, "horizontal", mask, 0);

    ASSERT_EQ((int)s2v_transforms.size(), sz);

    // Transforms should be identity or near-identity (insufficient data path)
    for (int k = 0; k < sz; k++) {
        auto params = s2v_transforms[k]->GetParameters();
        // Translation should be small
        ASSERT_TRUE(std::fabs(params[0]) < 5.0);
        ASSERT_TRUE(std::fabs(params[1]) < 5.0);
        ASSERT_TRUE(std::fabs(params[2]) < 5.0);
    }
}


// ============================================================
// Test: do_eddy flag — eddy params stay zero when disabled
// ============================================================
void test_s2v_do_eddy_flag()
{
    int sx = 16, sy = 16, sz = 8;
    auto img = create_s2v_test_image(sx, sy, sz);
    auto mask = create_block_mask(sx, sy, sz, 2.0, 2.0, 2.0, 1);
    auto lim_arr = compute_lim_arr(img, img);

    // Use MB>1 path (do_eddy is used in the flag setup for both paths,
    // but MB>1 path is more direct for testing)
    auto slspec = make_slspec_mb2(sz);

    // Run with do_eddy=false
    std::vector<TransformType::Pointer> s2v_noeddy;
    VolumeToSliceRegistration(img, img, slspec, lim_arr,
                              s2v_noeddy, false, "horizontal", mask, 0);

    // Run with do_eddy=true (identical images, so eddy should still be ~0)
    std::vector<TransformType::Pointer> s2v_eddy;
    VolumeToSliceRegistration(img, img, slspec, lim_arr,
                              s2v_eddy, true, "horizontal", mask, 0);

    ASSERT_EQ((int)s2v_noeddy.size(), sz);
    ASSERT_EQ((int)s2v_eddy.size(), sz);

    // With do_eddy=false, eddy parameters (6-13) should remain near-zero
    // (they're not optimized, so they stay at initial identity values)
    for (int k = 0; k < sz; k++) {
        auto p = s2v_noeddy[k]->GetParameters();
        // params[6] = 1 at identity for "horizontal" phase
        // params[7-13] should be 0
        for (int pp = 7; pp <= 13; pp++) {
            ASSERT_NEAR(p[pp], 0.0, 0.01);
        }
    }

    // With do_eddy=true but identical images, eddy parameters should
    // also be near their identity values (no actual eddy to recover)
    for (int k = 0; k < sz; k++) {
        auto p = s2v_eddy[k]->GetParameters();
        // The linear eddy terms (6-11) and quadratic (12-13) should be small
        // except param[6] which is 1 at identity for "horizontal"
        for (int pp = 7; pp <= 13; pp++) {
            ASSERT_TRUE(std::fabs(p[pp]) < 1.0);
        }
    }
}


int main()
{
    std::cout << "=== S2V Registration Tests ===" << std::endl;

    TEST(s2v_identity_no_motion);
    TEST(s2v_recovers_per_slice_shift);
    TEST(s2v_multiband_grouping);
    TEST(s2v_insufficient_voxels);
    TEST(s2v_do_eddy_flag);

    TEST_SUMMARY();
}
