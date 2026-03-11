// Tests for the transform <-> displacement field round-trip:
// ConvertEddyTransformToField and InvertDisplacementFieldImageFilterOkan.

#include "test_macros.h"
#include "test_image_helpers.h"
#include "convert_eddy_trans_to_field.hxx"
#include "itkInvertDisplacementFieldImageFilterOkan.h"
#include <cmath>


// --- Helper: create a test image with identity direction and centered origin ---
static ImageType3D::Pointer create_centered_image(int sx, int sy, int sz,
                                                    double sp = 2.0)
{
    auto img = ImageType3D::New();
    ImageType3D::SizeType size;
    size[0] = sx; size[1] = sy; size[2] = sz;
    ImageType3D::IndexType start; start.Fill(0);
    ImageType3D::RegionType region; region.SetSize(size); region.SetIndex(start);
    ImageType3D::SpacingType spacing;
    spacing[0] = sp; spacing[1] = sp; spacing[2] = sp;
    ImageType3D::DirectionType dir; dir.SetIdentity();

    // Center-voxel origin (same as DIFFPREP DP convention)
    ImageType3D::PointType origin;
    origin[0] = -((int)(sx) - 1) / 2.0 * sp;
    origin[1] = -((int)(sy) - 1) / 2.0 * sp;
    origin[2] = -((int)(sz) - 1) / 2.0 * sp;

    img->SetRegions(region);
    img->SetSpacing(spacing);
    img->SetOrigin(origin);
    img->SetDirection(dir);
    img->Allocate();
    img->FillBuffer(0);

    return img;
}


// --- Helper: fill image with centered Gaussian blob ---
static void fill_gaussian(ImageType3D::Pointer img, float amplitude = 1000.0f,
                           double sigma = 8.0)
{
    auto sz = img->GetLargestPossibleRegion().GetSize();
    auto sp = img->GetSpacing();
    double cx = (sz[0] - 1) * sp[0] / 2.0 + img->GetOrigin()[0];
    double cy = (sz[1] - 1) * sp[1] / 2.0 + img->GetOrigin()[1];
    double cz = (sz[2] - 1) * sp[2] / 2.0 + img->GetOrigin()[2];

    itk::ImageRegionIteratorWithIndex<ImageType3D> it(img, img->GetLargestPossibleRegion());
    for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
        ImageType3D::PointType pt;
        img->TransformIndexToPhysicalPoint(it.GetIndex(), pt);
        double dx = (pt[0] - cx) / sigma;
        double dy = (pt[1] - cy) / sigma;
        double dz = (pt[2] - cz) / sigma;
        float val = amplitude * std::exp(-0.5 * (dx * dx + dy * dy + dz * dz));
        it.Set(val);
    }
}


// --- Helper: compute max displacement magnitude in a field ---
static double max_displacement_magnitude(DisplacementFieldType::Pointer field)
{
    double max_mag = 0;
    itk::ImageRegionIteratorWithIndex<DisplacementFieldType> it(field, field->GetLargestPossibleRegion());
    for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
        auto vec = it.Get();
        double mag = std::sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
        if (mag > max_mag) max_mag = mag;
    }
    return max_mag;
}


// --- Helper: compute mean displacement vector in a field ---
static void mean_displacement(DisplacementFieldType::Pointer field,
                               double &mx, double &my, double &mz)
{
    mx = my = mz = 0;
    long count = 0;
    itk::ImageRegionIteratorWithIndex<DisplacementFieldType> it(field, field->GetLargestPossibleRegion());
    for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
        auto vec = it.Get();
        mx += vec[0]; my += vec[1]; mz += vec[2];
        count++;
    }
    if (count > 0) { mx /= count; my /= count; mz /= count; }
}


// ============================================================
// Test: identity transform gives zero displacement field
// ============================================================
void test_identity_transform_gives_zero_field()
{
    auto ref_img = create_centered_image(24, 24, 12);
    auto ref_img_DP = ref_img;  // Same for identity direction

    auto transform = OkanQuadraticTransformType::New();
    transform->SetPhase("horizontal");
    transform->SetIdentity();
    auto composite = wrap_in_composite(transform);

    auto field = ConvertEddyTransformToField(composite, ref_img, ref_img_DP);

    // All displacement vectors should be (0,0,0)
    double max_mag = max_displacement_magnitude(field);
    ASSERT_NEAR(max_mag, 0.0, 1e-8);
}


// ============================================================
// Test: translation field roundtrip (forward + invert)
// ============================================================
void test_translation_field_roundtrip()
{
    auto ref_img = create_centered_image(24, 24, 12);
    auto ref_img_DP = ref_img;

    // Create transform with 2mm X translation
    auto transform = OkanQuadraticTransformType::New();
    transform->SetPhase("horizontal");
    transform->SetIdentity();
    auto params = transform->GetParameters();
    params[0] = 2.0;  // 2mm X translation
    transform->SetParameters(params);
    auto composite = wrap_in_composite(transform);

    // Convert to displacement field
    auto field = ConvertEddyTransformToField(composite, ref_img, ref_img_DP);

    // Verify mean displacement is approximately (2, 0, 0)
    double mx, my, mz;
    mean_displacement(field, mx, my, mz);
    ASSERT_NEAR(mx, 2.0, 0.01);
    ASSERT_NEAR(my, 0.0, 0.01);
    ASSERT_NEAR(mz, 0.0, 0.01);

    // Invert the field
    using InverterType = itk::InvertDisplacementFieldImageFilterOkan<DisplacementFieldType>;
    auto inverter = InverterType::New();
    inverter->SetInput(field);
    inverter->SetMaximumNumberOfIterations(50);
    inverter->SetMeanErrorToleranceThreshold(0.0004);
    inverter->SetMaxErrorToleranceThreshold(0.04);
    inverter->SetNumberOfWorkUnits(1);
    inverter->Update();
    auto inv_field = inverter->GetOutput();

    // Inverted field should have approximately (-2, 0, 0) in the interior.
    // Boundary voxels are forced to zero by the inversion filter,
    // so check a single interior voxel instead of the full mean.
    ImageType3D::IndexType interior_idx;
    interior_idx[0] = 12; interior_idx[1] = 12; interior_idx[2] = 6;
    auto inv_vec = inv_field->GetPixel(interior_idx);
    ASSERT_NEAR(inv_vec[0], -2.0, 0.3);
    ASSERT_NEAR(inv_vec[1], 0.0, 0.1);
    ASSERT_NEAR(inv_vec[2], 0.0, 0.1);
}


// ============================================================
// Test: rotation field roundtrip
// ============================================================
void test_rotation_field_roundtrip()
{
    auto ref_img = create_centered_image(24, 24, 12);
    auto ref_img_DP = ref_img;

    // Create transform with small rotation (0.02 rad ~ 1.15 degrees)
    auto transform = OkanQuadraticTransformType::New();
    transform->SetPhase("horizontal");
    transform->SetIdentity();
    auto params = transform->GetParameters();
    params[5] = 0.02;  // Z-axis rotation
    transform->SetParameters(params);
    auto composite = wrap_in_composite(transform);

    // Convert to displacement field
    auto field = ConvertEddyTransformToField(composite, ref_img, ref_img_DP);

    // Rotation field should be nonzero (except at center)
    double max_mag = max_displacement_magnitude(field);
    ASSERT_TRUE(max_mag > 0.1);

    // Invert the field
    using InverterType = itk::InvertDisplacementFieldImageFilterOkan<DisplacementFieldType>;
    auto inverter = InverterType::New();
    inverter->SetInput(field);
    inverter->SetMaximumNumberOfIterations(50);
    inverter->SetMeanErrorToleranceThreshold(0.0004);
    inverter->SetMaxErrorToleranceThreshold(0.04);
    inverter->SetNumberOfWorkUnits(1);
    inverter->Update();
    auto inv_field = inverter->GetOutput();

    // Apply forward then inverse to a test point — should return near original.
    // Use the center of the image as a reference: displacement should be ~0 at center.
    // Use an off-center point to test the round-trip.
    ImageType3D::IndexType test_idx;
    test_idx[0] = 16; test_idx[1] = 16; test_idx[2] = 6;
    ImageType3D::PointType test_pt;
    ref_img->TransformIndexToPhysicalPoint(test_idx, test_pt);

    // Forward displacement at test point
    auto fwd_disp = field->GetPixel(test_idx);
    ImageType3D::PointType fwd_pt;
    fwd_pt[0] = test_pt[0] + fwd_disp[0];
    fwd_pt[1] = test_pt[1] + fwd_disp[1];
    fwd_pt[2] = test_pt[2] + fwd_disp[2];

    // Get inverse displacement at the forward point (nearest voxel)
    itk::ContinuousIndex<double, 3> fwd_cind;
    inv_field->TransformPhysicalPointToContinuousIndex(fwd_pt, fwd_cind);
    ImageType3D::IndexType fwd_idx;
    for (int d = 0; d < 3; d++)
        fwd_idx[d] = (int)std::round(fwd_cind[d]);

    if (inv_field->GetLargestPossibleRegion().IsInside(fwd_idx)) {
        auto inv_disp = inv_field->GetPixel(fwd_idx);
        ImageType3D::PointType round_trip_pt;
        round_trip_pt[0] = fwd_pt[0] + inv_disp[0];
        round_trip_pt[1] = fwd_pt[1] + inv_disp[1];
        round_trip_pt[2] = fwd_pt[2] + inv_disp[2];

        // Round-trip should return near the original point
        double err = std::sqrt(
            (round_trip_pt[0] - test_pt[0]) * (round_trip_pt[0] - test_pt[0]) +
            (round_trip_pt[1] - test_pt[1]) * (round_trip_pt[1] - test_pt[1]) +
            (round_trip_pt[2] - test_pt[2]) * (round_trip_pt[2] - test_pt[2]));
        ASSERT_TRUE(err < 0.5);  // Within half a voxel
    }
}


// ============================================================
// Test: eddy field inversion accuracy with quadratic terms
// ============================================================
void test_eddy_field_inversion_accuracy()
{
    auto ref_img = create_centered_image(24, 24, 12);
    fill_gaussian(ref_img);
    auto ref_img_DP = ref_img;

    // Create transform with quadratic eddy terms
    auto transform = OkanQuadraticTransformType::New();
    transform->SetPhase("horizontal");
    transform->SetIdentity();
    auto params = transform->GetParameters();
    params[12] = 0.0005;   // Quadratic eddy term
    params[13] = 0.0003;   // Quadratic eddy term
    transform->SetParameters(params);
    auto composite = wrap_in_composite(transform);

    // Convert to field
    auto field = ConvertEddyTransformToField(composite, ref_img, ref_img_DP);

    // Field should be nonzero (quadratic terms cause position-dependent displacement)
    double max_mag = max_displacement_magnitude(field);
    ASSERT_TRUE(max_mag > 0.01);

    // Invert the field
    using InverterType = itk::InvertDisplacementFieldImageFilterOkan<DisplacementFieldType>;
    auto inverter = InverterType::New();
    inverter->SetInput(field);
    inverter->SetMaximumNumberOfIterations(50);
    inverter->SetMeanErrorToleranceThreshold(0.0004);
    inverter->SetMaxErrorToleranceThreshold(0.04);
    inverter->SetNumberOfWorkUnits(1);
    inverter->Update();
    auto inv_field = inverter->GetOutput();

    // Resample the Gaussian blob through the forward field then inverted field
    // and compare to original. For a pure translation test point:
    // Check that interior voxels of the round-trip match within tolerance
    using DisplacementFieldTransformType = itk::DisplacementFieldTransform<double, 3>;
    auto fwd_trans = DisplacementFieldTransformType::New();
    fwd_trans->SetDisplacementField(field);
    auto inv_trans = DisplacementFieldTransformType::New();
    inv_trans->SetDisplacementField(inv_field);

    // Sample a few interior points and check round-trip accuracy
    int good_points = 0, total_points = 0;
    ImageType3D::IndexType idx;
    for (int k = 3; k < 9; k++) {
        idx[2] = k;
        for (int j = 6; j < 18; j++) {
            idx[1] = j;
            for (int i = 6; i < 18; i++) {
                idx[0] = i;
                ImageType3D::PointType pt;
                ref_img->TransformIndexToPhysicalPoint(idx, pt);

                // Forward
                ImageType3D::PointType fwd_pt = fwd_trans->TransformPoint(pt);
                // Inverse
                ImageType3D::PointType rt_pt = inv_trans->TransformPoint(fwd_pt);

                double err = std::sqrt(
                    (rt_pt[0] - pt[0]) * (rt_pt[0] - pt[0]) +
                    (rt_pt[1] - pt[1]) * (rt_pt[1] - pt[1]) +
                    (rt_pt[2] - pt[2]) * (rt_pt[2] - pt[2]));

                total_points++;
                if (err < 0.5)  // Within half a voxel
                    good_points++;
            }
        }
    }

    // At least 90% of interior points should round-trip accurately
    ASSERT_TRUE(total_points > 0);
    ASSERT_TRUE((double)good_points / total_points > 0.9);
}


// ============================================================
// Test: DP header conversion (oblique direction -> identity)
// ============================================================
void test_dp_header_conversion()
{
    // Create an image with non-identity direction (oblique acquisition)
    auto img = create_centered_image(24, 24, 12);
    ImageType3D::DirectionType oblique_dir;
    oblique_dir.SetIdentity();
    // Small rotation around Z axis (5 degrees)
    double angle = 5.0 * M_PI / 180.0;
    oblique_dir[0][0] = cos(angle);  oblique_dir[0][1] = -sin(angle);
    oblique_dir[1][0] = sin(angle);  oblique_dir[1][1] = cos(angle);
    img->SetDirection(oblique_dir);

    // Apply center-voxel DP conversion
    auto dp_img = make_dp_image(img);

    // Direction should be identity
    auto dp_dir = dp_img->GetDirection();
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            ASSERT_NEAR(dp_dir[i][j], (i == j) ? 1.0 : 0.0, 1e-10);

    // Origin should be at center-voxel
    auto sz = img->GetLargestPossibleRegion().GetSize();
    auto sp = img->GetSpacing();
    ASSERT_NEAR(dp_img->GetOrigin()[0], -((int)(sz[0]) - 1) / 2.0 * sp[0], 1e-10);
    ASSERT_NEAR(dp_img->GetOrigin()[1], -((int)(sz[1]) - 1) / 2.0 * sp[1], 1e-10);
    ASSERT_NEAR(dp_img->GetOrigin()[2], -((int)(sz[2]) - 1) / 2.0 * sp[2], 1e-10);

    // Spacing should be preserved
    ASSERT_NEAR(dp_img->GetSpacing()[0], sp[0], 1e-10);
    ASSERT_NEAR(dp_img->GetSpacing()[1], sp[1], 1e-10);
    ASSERT_NEAR(dp_img->GetSpacing()[2], sp[2], 1e-10);
}


int main()
{
    std::cout << "=== Displacement Field Tests ===" << std::endl;

    TEST(identity_transform_gives_zero_field);
    TEST(translation_field_roundtrip);
    TEST(rotation_field_roundtrip);
    TEST(eddy_field_inversion_accuracy);
    TEST(dp_header_conversion);

    TEST_SUMMARY();
}
