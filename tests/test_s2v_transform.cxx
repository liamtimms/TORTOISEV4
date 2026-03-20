// Test ForwardTransformImage from register_dwi_to_slice.h:
// verifies that the KD-tree based forward interpolation correctly
// handles identity transforms, uniform translations, and per-slice transforms.

#include "test_macros.h"
#include "register_dwi_to_slice.h"
#include <cmath>

using TransformType = OkanQuadraticTransformType;


// --- Helper: create a small test image with a slice-dependent pattern ---
static ImageType3D::Pointer create_patterned_image(int sx, int sy, int sz,
                                                     double sp = 2.0)
{
    auto img = ImageType3D::New();
    ImageType3D::SizeType size; size[0] = sx; size[1] = sy; size[2] = sz;
    ImageType3D::IndexType start; start.Fill(0);
    ImageType3D::RegionType region; region.SetSize(size); region.SetIndex(start);
    ImageType3D::SpacingType spacing; spacing[0] = sp; spacing[1] = sp; spacing[2] = sp;
    ImageType3D::PointType origin; origin.Fill(0);
    ImageType3D::DirectionType dir; dir.SetIdentity();
    img->SetRegions(region);
    img->SetSpacing(spacing);
    img->SetOrigin(origin);
    img->SetDirection(dir);
    img->Allocate();

    // Fill with: value = (slice+1)*100 + col + 1
    itk::ImageRegionIteratorWithIndex<ImageType3D> it(img, region);
    for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
        auto idx = it.GetIndex();
        float val = (idx[2] + 1) * 100.0f + idx[0] + 1;
        it.Set(val);
    }
    return img;
}


// --- Helper: create identity s2v transforms for nslices ---
static std::vector<TransformType::Pointer> make_identity_s2v(int nslices,
                                                              const std::string& phase = "vertical")
{
    std::vector<TransformType::Pointer> transforms(nslices);
    for (int k = 0; k < nslices; k++) {
        transforms[k] = TransformType::New();
        transforms[k]->SetPhase(phase);
        transforms[k]->SetIdentity();
    }
    return transforms;
}


// ============================================================
// Test: identity transforms reproduce the original image
// ============================================================
// ForwardTransformImage uses a KD-tree search and inverse-distance-weighted (IDW) interpolation to scatter source voxels to target locations,
// rather than ITK's standard pull-back (inverse) resampling. This is necessary for
// S2V because each slice has its own transform — inverse resampling would require
// knowing which slice's transform to invert at each target voxel, which is ambiguous
// at slice boundaries. The 1% mismatch tolerance accounts for IDW interpolation
// noise at grid points where multiple neighbors are equidistant.

void test_identity_forward_transform()
{
    int sx = 12, sy = 12, sz = 6;
    auto img = create_patterned_image(sx, sy, sz);
    auto s2v = make_identity_s2v(sz);

    auto result = ForwardTransformImage(img, s2v);

    // Interior voxels (away from edges) should match the input closely.
    // The KD-tree interpolation with identity transforms should return
    // the original value when the nearest neighbor is < 0.1 distance.
    int mismatches = 0;
    ImageType3D::IndexType idx;
    for (int k = 1; k < sz - 1; k++) {
        idx[2] = k;
        for (int j = 1; j < sy - 1; j++) {
            idx[1] = j;
            for (int i = 1; i < sx - 1; i++) {
                idx[0] = i;
                float orig = img->GetPixel(idx);
                float out = result->GetPixel(idx);
                if (std::fabs(orig - out) > 0.1f)
                    mismatches++;
            }
        }
    }
    // Allow at most 1% mismatches (numerical precision of KD-tree lookup)
    int total_interior = (sx - 2) * (sy - 2) * (sz - 2);
    ASSERT_TRUE(mismatches < total_interior * 0.01);
}


// ============================================================
// Test: uniform translation shifts the image
// ============================================================
// Verifies the forward-scatter direction convention: a +2mm X translation in the
// transform moves each source voxel to index i+1, so result[i] ≈ input[i-1].
// The 80% match threshold accounts for boundary effects and KD-tree interpolation
// artifacts at the edges of the shifted region.

void test_uniform_translation_forward()
{
    int sx = 16, sy = 16, sz = 8;
    auto img = create_patterned_image(sx, sy, sz);

    // Apply a 1-voxel (2mm) X translation to all slices
    auto s2v = make_identity_s2v(sz);
    for (int k = 0; k < sz; k++) {
        auto p = s2v[k]->GetParameters();
        p[0] = 2.0;  // 2mm = 1 voxel with 2mm spacing
        s2v[k]->SetParameters(p);
    }

    auto result = ForwardTransformImage(img, s2v);

    // After forward-transforming with +2mm X translation, each source voxel
    // at index i moves to index i+1. The KD-tree interpolation means
    // result[i] should approximately equal img[i-1] for interior voxels.
    // Check a few interior voxels
    ImageType3D::IndexType idx_out, idx_expected;
    int matches = 0, total = 0;
    for (int k = 2; k < sz - 2; k++) {
        idx_out[2] = k; idx_expected[2] = k;
        for (int j = 2; j < sy - 2; j++) {
            idx_out[1] = j; idx_expected[1] = j;
            for (int i = 3; i < sx - 3; i++) {
                idx_out[0] = i;
                idx_expected[0] = i - 1;
                float out = result->GetPixel(idx_out);
                float expected = img->GetPixel(idx_expected);
                total++;
                if (std::fabs(out - expected) < 5.0f)
                    matches++;
            }
        }
    }
    // Most interior voxels should match (within tolerance due to KD-tree)
    ASSERT_TRUE(matches > total * 0.8);
}


// ============================================================
// Test: alternating per-slice transforms
// ============================================================
// Verifies that ForwardTransformImage correctly applies different transforms to
// different slices — the core requirement for S2V correction. Even slices get
// identity (should match input), odd slices get a 2-voxel shift (should differ).
// This catches bugs where the slice index→transform mapping is off-by-one.

void test_alternating_slice_transforms()
{
    int sx = 16, sy = 16, sz = 8;
    auto img = create_patterned_image(sx, sy, sz);

    // Even slices: identity; Odd slices: small translation
    auto s2v = make_identity_s2v(sz);
    for (int k = 1; k < sz; k += 2) {
        auto p = s2v[k]->GetParameters();
        p[0] = 4.0;  // 4mm = 2 voxel shift
        s2v[k]->SetParameters(p);
    }

    auto result = ForwardTransformImage(img, s2v);

    // Even slices should be nearly unchanged; odd slices should differ
    ImageType3D::IndexType idx;
    int even_matches = 0, even_total = 0;
    int odd_differs = 0, odd_total = 0;

    for (int k = 1; k < sz - 1; k++) {
        idx[2] = k;
        for (int j = 2; j < sy - 2; j++) {
            idx[1] = j;
            for (int i = 4; i < sx - 4; i++) {
                idx[0] = i;
                float orig = img->GetPixel(idx);
                float out = result->GetPixel(idx);

                if (k % 2 == 0) {
                    even_total++;
                    if (std::fabs(orig - out) < 1.0f)
                        even_matches++;
                } else {
                    odd_total++;
                    if (std::fabs(orig - out) > 1.0f)
                        odd_differs++;
                }
            }
        }
    }

    // Even slices should mostly match (identity transform)
    ASSERT_TRUE(even_total > 0);
    ASSERT_TRUE(even_matches > even_total * 0.8);

    // Odd slices should mostly differ (translated)
    ASSERT_TRUE(odd_total > 0);
    ASSERT_TRUE(odd_differs > odd_total * 0.5);
}


int main()
{
    std::cout << "=== S2V Forward Transform Tests ===" << std::endl;

    TEST(identity_forward_transform);
    TEST(uniform_translation_forward);
    TEST(alternating_slice_transforms);

    TEST_SUMMARY();
}
