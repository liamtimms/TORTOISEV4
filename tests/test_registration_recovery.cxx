// End-to-end test of RegisterDWIToB0: verifies that volume-to-volume
// registration recovers known translations, rotations, and eddy distortions
// from synthetic Gaussian blob images.

#include "test_macros.h"
#include "test_image_helpers.h"
#include "register_dwi_to_b0.h"
#include "rigid_register_images.h"

// RegisterDWIToB0 calls TORTOISE::GetAvailableITKThreadFor() which needs
// these static definitions. Provided via linking TORTOISE_global.cxx.


// --- Helper: create a MeccSettings with specified flags ---
static MeccSettings make_rigid_settings()
{
    MeccSettings s;
    // Default MeccSettings already has flags[0-5]=1 (rigid), rest=0
    return s;
}

static MeccSettings make_quadratic_settings()
{
    MeccSettings s;
    auto flags = s.getFlags();
    // Enable quadratic eddy terms (params 6-13)
    for (int i = 6; i <= 13; i++)
        flags[i] = true;
    s.setFlags(flags);
    return s;
}


// ============================================================
// Test: recover a known translation
// ============================================================
// Verifies RegisterDWIToB0 recovers a known 2/1.5/1mm translation from a synthetic
// Gaussian blob. This is the core of DIFFPREP's per-volume motion correction.
// 0.5mm tolerance is ~25% of voxel size (2mm) — tighter than clinical requirements
// but loose enough for a 48³ synthetic image with limited spatial frequency content.
// Recovered parameters have opposite signs because RegisterDWIToB0 finds the
// moving→fixed (inverse) transform.

void test_translation_recovery()
{
    // Create fixed image: 48x48x24, 2mm isotropic, Gaussian blob
    auto fixed = create_gaussian_blob(48, 48, 24, 2.0, 2.0, 2.0,
                                       20.0, 25.0, 15.0, 800.0f);

    // Create displaced moving image: apply known translation
    auto true_transform = TestTransformType::New();
    true_transform->SetPhase("vertical");
    true_transform->SetIdentity();
    auto tp = true_transform->GetParameters();
    tp[0] = 2.0;   // 2mm X shift
    tp[1] = -1.5;  // 1.5mm Y shift
    tp[2] = 1.0;   // 1mm Z shift
    true_transform->SetParameters(tp);

    auto moving = apply_transform_to_image(fixed, true_transform);

    // Compute intensity limits
    auto frange = compute_image_range(fixed);
    auto mrange = compute_image_range(moving);
    std::vector<float> lim_arr = {frange.first, frange.second, mrange.first, mrange.second};

    // Run registration
    MeccSettings settings = make_rigid_settings();
    auto result = RegisterDWIToB0(fixed, moving, "vertical", &settings,
                                   true, lim_arr, 0);

    ASSERT_TRUE(result != nullptr);

    auto rp = result->GetParameters();

    // RegisterDWIToB0 finds the transform that maps moving→fixed,
    // which is the inverse of the applied transform. So recovered
    // parameters should have opposite signs.
    // Allow 0.5mm tolerance for translation on this image size.
    ASSERT_NEAR(rp[0], -tp[0], 0.5);
    ASSERT_NEAR(rp[1], -tp[1], 0.5);
    ASSERT_NEAR(rp[2], -tp[2], 0.5);

    // Rotation should remain near zero
    ASSERT_NEAR(rp[3], 0.0, 0.01);
    ASSERT_NEAR(rp[4], 0.0, 0.01);
    ASSERT_NEAR(rp[5], 0.0, 0.01);
}


// ============================================================
// Test: recover a rigid transform (translation + rotation)
// ============================================================
// Tests coupled translation+rotation recovery. Wider tolerance (1.0mm, 0.02 rad)
// because translation and rotation are coupled in the optimization — a small
// rotation error gets absorbed into translation and vice versa, especially with
// the smooth Gaussian blob providing limited rotational gradient information.

void test_rigid_recovery()
{
    auto fixed = create_gaussian_blob(48, 48, 24, 2.0, 2.0, 2.0,
                                       20.0, 25.0, 15.0, 800.0f);

    auto true_transform = TestTransformType::New();
    true_transform->SetPhase("vertical");
    true_transform->SetIdentity();
    auto tp = true_transform->GetParameters();
    tp[0] = 1.0;     // X translation
    tp[1] = -1.0;    // Y translation
    tp[3] = 0.02;    // X rotation (radians)
    tp[5] = -0.015;  // Z rotation
    true_transform->SetParameters(tp);

    auto moving = apply_transform_to_image(fixed, true_transform);

    auto frange2 = compute_image_range(fixed);
    auto mrange2 = compute_image_range(moving);
    std::vector<float> lim_arr = {frange2.first, frange2.second, mrange2.first, mrange2.second};

    MeccSettings settings = make_rigid_settings();
    auto result = RegisterDWIToB0(fixed, moving, "vertical", &settings,
                                   true, lim_arr, 0);

    ASSERT_TRUE(result != nullptr);
    auto rp = result->GetParameters();

    // Inverse convention: recovered params have opposite signs
    // Wider tolerance for rigid (translation+rotation coupled optimization)
    ASSERT_NEAR(rp[0], -tp[0], 1.0);
    ASSERT_NEAR(rp[1], -tp[1], 1.0);
    ASSERT_NEAR(rp[3], -tp[3], 0.02);
    ASSERT_NEAR(rp[5], -tp[5], 0.02);
}


// ============================================================
// Test: optimization flags are respected (rotation disabled)
// ============================================================
// Verifies that MeccSettings flags correctly gate which parameters the optimizer
// can modify. The "justeddy" MECC profile disables rotation (flags[3-5]=false)
// to speed up registration when motion is known to be small. If flags are ignored,
// the optimizer wastes iterations on unnecessary DOFs.

void test_flags_respected()
{
    auto fixed = create_gaussian_blob(48, 48, 24, 2.0, 2.0, 2.0,
                                       20.0, 25.0, 15.0, 800.0f);

    // Apply both translation AND rotation
    auto true_transform = TestTransformType::New();
    true_transform->SetPhase("vertical");
    true_transform->SetIdentity();
    auto tp = true_transform->GetParameters();
    tp[0] = 2.0;
    tp[3] = 0.03;  // rotation that we'll prevent recovery of
    true_transform->SetParameters(tp);

    auto moving = apply_transform_to_image(fixed, true_transform);

    auto frange3 = compute_image_range(fixed);
    auto mrange3 = compute_image_range(moving);
    std::vector<float> lim_arr = {frange3.first, frange3.second, mrange3.first, mrange3.second};

    // Disable rotation flags
    MeccSettings settings;
    auto flags = settings.getFlags();
    flags[3] = false; flags[4] = false; flags[5] = false;
    settings.setFlags(flags);

    auto result = RegisterDWIToB0(fixed, moving, "vertical", &settings,
                                   true, lim_arr, 0);

    ASSERT_TRUE(result != nullptr);
    auto rp = result->GetParameters();

    // Rotation params should stay at zero since they're disabled
    ASSERT_NEAR(rp[3], 0.0, 1e-10);
    ASSERT_NEAR(rp[4], 0.0, 1e-10);
    ASSERT_NEAR(rp[5], 0.0, 1e-10);
}


// ============================================================
// Test: MultiStartRigidSearchCoarseToFine recovers large rotation
// ============================================================
// Tests the multi-resolution initialization used when --large_motion_correction=1.
// A 25° rotation exceeds the capture range of single-scale registration, so the
// coarse-to-fine search (4x→2x downsampling with multi-start) is required.
// 15° tolerance (0.26 rad) is generous because the coarse search uses 45° angular
// steps — we only need the correct basin, not sub-degree precision.

void test_coarse_to_fine_large_rotation()
{
    // Create fixed: large enough for downsampling (4x and 2x)
    auto fixed = create_gaussian_blob(48, 48, 24, 2.0, 2.0, 2.0,
                                       20.0, 25.0, 15.0, 800.0f);

    // Apply 25° rotation about Y (0.436 rad)
    auto true_transform = TestTransformType::New();
    true_transform->SetPhase("vertical");
    true_transform->SetIdentity();
    auto tp = true_transform->GetParameters();
    tp[4] = 0.436;  // ~25° about Y
    true_transform->SetParameters(tp);

    auto moving = apply_transform_to_image(fixed, true_transform);

    // Call coarse-to-fine multistart search
    auto result = MultiStartRigidSearchCoarseToFine(fixed, moving, "CC");

    ASSERT_TRUE(result != nullptr);

    auto rp = result->GetParameters();

    // Euler3DTransform params: [Rx, Ry, Rz, Tx, Ty, Tz]
    // OkanQuadratic params:    [Tx, Ty, Tz, Rx, Ry, Rz, ...]
    // We applied Ry=0.436 in Okan space (index 4), recover from Euler (index 1)
    double recovered_ry = rp[1];
    // Allow 15° tolerance (0.26 rad) — coarse search uses large steps
    ASSERT_TRUE(std::fabs(std::fabs(recovered_ry) - 0.436) < 0.26);
}


int main()
{
    std::cout << "=== Registration Recovery Tests ===" << std::endl;

    TEST(translation_recovery);
    TEST(rigid_recovery);
    TEST(flags_respected);
    TEST(coarse_to_fine_large_rotation);

    TEST_SUMMARY();
}
