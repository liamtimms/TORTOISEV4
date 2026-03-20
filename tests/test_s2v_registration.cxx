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
// Baseline sanity: VolumeToSliceRegistration with identical images should return
// near-identity transforms. Failure means the registration has a systematic bias
// or the metric is not correctly minimized at identity.
// Wider Z tolerance (1.5mm vs 0.5mm in-plane) because the 2-slice window approach
// doubles the through-plane spacing, reducing Z-axis constraint.

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
// Core S2V use case: individual slices can move independently during acquisition.
// Slices 5-7 are shifted by 1 voxel (2mm) in X. The S2V loop in DIFFPREP.cxx
// (DIFFPREP::MotionAndEddy) calls VolumeToSliceRegistration per epoch to recover these
// per-slice motions. Detection threshold of 0.3mm is well below voxel size.

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
// In multiband (MB) acquisitions, multiple slices are excited simultaneously and
// share the same motion state. The slspec matrix encodes which slices belong to
// each excitation group. VolumeToSliceRegistration must assign identical transforms
// to all slices in the same group — otherwise, simultaneously-acquired slices get
// different corrections, creating discontinuities in the corrected volume.

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
// Edge case guard: slices at the top/bottom of the brain mask may have very few
// nonzero voxels (< 10% of slice area). The registration must degrade gracefully
// to identity rather than crashing on a singular metric matrix. This happens
// regularly with tight brain masks on the first/last few slices.

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
// When do_eddy=false, only rigid parameters (0-5) are optimized at the slice level.
// Eddy parameters (6-13) must remain at their identity values. This matches
// DIFFPREP's behavior where volume-level eddy correction handles eddy currents
// and S2V only handles intra-volume motion.

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


// ============================================================
// Test: warm_start preserves pre-existing transforms
// ============================================================
// Warm start initializes from the previous epoch's transforms instead of identity.
// This is analogous to FSL eddy's --s2v_niter: each sub-iteration refines the
// previous result. Without warm start, later epochs discard earlier progress and
// restart from scratch, wasting computation and potentially losing convergence.

void test_s2v_warm_start_preserves()
{
    int sx = 16, sy = 16, sz = 8;
    auto img = create_s2v_test_image(sx, sy, sz);
    auto mask = create_block_mask(sx, sy, sz, 2.0, 2.0, 2.0, 1);
    auto lim_arr = compute_lim_arr(img, img);
    auto slspec = make_slspec_mb1(sz);

    // Pre-fill transforms with known Tz = 1mm on each slice
    std::vector<TransformType::Pointer> warm_transforms(sz);
    for (int k = 0; k < sz; k++) {
        warm_transforms[k] = TransformType::New();
        warm_transforms[k]->SetPhase("horizontal");
        warm_transforms[k]->SetIdentity();
        auto p = warm_transforms[k]->GetParameters();
        p[2] = 1.0;  // 1mm Z translation
        warm_transforms[k]->SetParameters(p);
    }

    // With warm_start=true on identical images, transforms should stay near initial
    VolumeToSliceRegistration(img, img, slspec, lim_arr,
                              warm_transforms, false, "horizontal", mask, 0, true, 0.0f);

    // Transforms should NOT have been reset to identity
    bool any_nonzero_tz = false;
    for (int k = 0; k < sz; k++) {
        auto p = warm_transforms[k]->GetParameters();
        if (std::fabs(p[2]) > 0.3)
            any_nonzero_tz = true;
    }
    ASSERT_TRUE(any_nonzero_tz);

    // Now with warm_start=false — transforms should be reset to identity
    std::vector<TransformType::Pointer> cold_transforms(sz);
    for (int k = 0; k < sz; k++) {
        cold_transforms[k] = TransformType::New();
        cold_transforms[k]->SetPhase("horizontal");
        cold_transforms[k]->SetIdentity();
        auto p = cold_transforms[k]->GetParameters();
        p[2] = 1.0;
        cold_transforms[k]->SetParameters(p);
    }

    VolumeToSliceRegistration(img, img, slspec, lim_arr,
                              cold_transforms, false, "horizontal", mask, 0, false, 0.0f);

    // With identical images and cold start, should be near-identity
    for (int k = 0; k < sz; k++) {
        auto p = cold_transforms[k]->GetParameters();
        ASSERT_NEAR(p[0], 0.0, 0.5);
        ASSERT_NEAR(p[1], 0.0, 0.5);
    }
}


// ============================================================
// Test: smoothing_sigma > 0 runs the two-level path without crashing
// ============================================================
// When smoothing_sigma > 0, the CPU path uses a two-level registration:
// first at the smoothed scale (wider capture range), then at full resolution.
// This is analogous to FSL eddy's --fwhm schedule. sigma=2.0 is a typical
// first-epoch value corresponding to ~4.7mm FWHM Gaussian smoothing.

void test_s2v_smoothing_sigma_bounded()
{
    int sx = 24, sy = 24, sz = 12;
    auto fixed = create_s2v_test_image(sx, sy, sz);
    auto mask = create_block_mask(sx, sy, sz, 2.0, 2.0, 2.0, 1);

    // Create moving with slices 5-7 shifted
    using DupType = itk::ImageDuplicator<ImageType3D>;
    auto dup = DupType::New();
    dup->SetInputImage(fixed);
    dup->Update();
    auto moving = dup->GetOutput();

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
            idx[0] = sx - 1;
            ImageType3D::IndexType prev_idx = idx;
            prev_idx[0] = sx - 2;
            moving->SetPixel(idx, fixed->GetPixel(prev_idx));
        }
    }

    auto lim_arr = compute_lim_arr(fixed, moving);
    auto slspec = make_slspec_mb1(sz);

    std::vector<TransformType::Pointer> s2v_transforms;

    // Call with smoothing_sigma=2.0 — this should exercise the two-level path
    VolumeToSliceRegistration(fixed, moving, slspec, lim_arr,
                              s2v_transforms, false, "horizontal", mask, 0, false, 2.0f);

    ASSERT_EQ((int)s2v_transforms.size(), sz);

    // All params should be finite
    for (int k = 0; k < sz; k++) {
        auto p = s2v_transforms[k]->GetParameters();
        for (unsigned int pp = 0; pp < p.GetSize(); pp++) {
            ASSERT_TRUE(std::isfinite(p[pp]));
        }
    }

    // At least one shifted slice should show detectable motion
    bool any_shift = false;
    for (int k = 5; k <= 7; k++) {
        auto p = s2v_transforms[k]->GetParameters();
        if (std::fabs(p[0]) > 0.2)
            any_shift = true;
    }
    ASSERT_TRUE(any_shift);
}


// ============================================================
// Test: VolumeToSliceRegistrationWithMultistart handles MB data
// ============================================================
// Tests the multi-start variant used when --s2v_multistart=1 for large intra-volume
// motion. Verifies it handles MB grouping correctly (same excitation → same params)
// and produces finite parameters. The 0.05 rad (~3°) rotation is within the
// multi-start search range but may exceed single-start capture range.

void test_s2v_multistart_mb()
{
    int sx = 16, sy = 16, sz = 8;
    auto fixed = create_s2v_test_image(sx, sy, sz);
    auto mask = create_block_mask(sx, sy, sz, 2.0, 2.0, 2.0, 1);

    // Create moving with a small rotation about Z (~0.05 rad ≈ 3°)
    auto rot_transform = TransformType::New();
    rot_transform->SetPhase("horizontal");
    rot_transform->SetIdentity();
    auto tp = rot_transform->GetParameters();
    tp[5] = 0.05;  // small Rz rotation
    rot_transform->SetParameters(tp);

    auto moving = apply_transform_to_image(fixed, rot_transform);

    auto lim_arr = compute_lim_arr(fixed, moving);
    auto slspec = make_slspec_mb2(sz);

    std::vector<TransformType::Pointer> s2v_transforms;

    // Call multistart variant — should complete without crash
    VolumeToSliceRegistrationWithMultistart(fixed, moving, slspec, lim_arr,
                                             s2v_transforms, false, "horizontal", mask, 0);

    ASSERT_EQ((int)s2v_transforms.size(), sz);

    // All transforms should be finite
    for (int k = 0; k < sz; k++) {
        auto p = s2v_transforms[k]->GetParameters();
        for (unsigned int pp = 0; pp < p.GetSize(); pp++) {
            ASSERT_TRUE(std::isfinite(p[pp]));
        }
    }

    // MB grouping should still hold: same excitation → same params
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


int main()
{
    std::cout << "=== S2V Registration Tests ===" << std::endl;

    TEST(s2v_identity_no_motion);
    TEST(s2v_recovers_per_slice_shift);
    TEST(s2v_multiband_grouping);
    TEST(s2v_insufficient_voxels);
    TEST(s2v_do_eddy_flag);
    TEST(s2v_warm_start_preserves);
    TEST(s2v_smoothing_sigma_bounded);
    TEST(s2v_multistart_mb);

    TEST_SUMMARY();
}
