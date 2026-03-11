#ifndef TEST_IMAGE_HELPERS_H
#define TEST_IMAGE_HELPERS_H

#include "itkImage.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkResampleImageFilter.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkImageDuplicator.h"
#include "itkCompositeTransform.h"
#include "itkOkanQuadraticTransform.h"
#include <cmath>

using TestImageType = itk::Image<float, 3>;
using TestTransformType = itk::OkanQuadraticTransform<double, 3, 3>;


// Create a 3D image with given dimensions, spacing, and fill value.
inline TestImageType::Pointer create_test_image(int sx, int sy, int sz,
                                                 double spx, double spy, double spz,
                                                 float fill_val = 0.0f)
{
    auto img = TestImageType::New();

    TestImageType::SizeType size;
    size[0] = sx; size[1] = sy; size[2] = sz;

    TestImageType::IndexType start;
    start.Fill(0);

    TestImageType::RegionType region;
    region.SetSize(size);
    region.SetIndex(start);

    TestImageType::SpacingType spacing;
    spacing[0] = spx; spacing[1] = spy; spacing[2] = spz;

    TestImageType::PointType origin;
    origin.Fill(0);

    TestImageType::DirectionType dir;
    dir.SetIdentity();

    img->SetRegions(region);
    img->SetSpacing(spacing);
    img->SetOrigin(origin);
    img->SetDirection(dir);
    img->Allocate();
    img->FillBuffer(fill_val);

    return img;
}


// Create a 3D image with a centered ellipsoidal Gaussian blob.
// Provides good MI contrast for registration tests.
inline TestImageType::Pointer create_gaussian_blob(int sx, int sy, int sz,
                                                    double spx, double spy, double spz,
                                                    double sigma_x, double sigma_y, double sigma_z,
                                                    float amplitude = 1000.0f)
{
    auto img = create_test_image(sx, sy, sz, spx, spy, spz, 0.0f);

    double cx = (sx - 1) * spx / 2.0;
    double cy = (sy - 1) * spy / 2.0;
    double cz = (sz - 1) * spz / 2.0;

    using IterType = itk::ImageRegionIteratorWithIndex<TestImageType>;
    IterType it(img, img->GetLargestPossibleRegion());
    for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
        TestImageType::PointType pt;
        img->TransformIndexToPhysicalPoint(it.GetIndex(), pt);

        double dx = (pt[0] - cx) / sigma_x;
        double dy = (pt[1] - cy) / sigma_y;
        double dz = (pt[2] - cz) / sigma_z;
        float val = amplitude * std::exp(-0.5 * (dx*dx + dy*dy + dz*dz));

        // Add some structure variation for MI (concentric rings)
        double r = std::sqrt(dx*dx + dy*dy + dz*dz);
        val *= (1.0f + 0.3f * std::sin(r * 3.0));

        it.Set(val);
    }

    return img;
}


// Apply an OkanQuadraticTransform to resample an image.
// Uses inverse mapping (standard ITK resampling).
inline TestImageType::Pointer apply_transform_to_image(
    TestImageType::Pointer img,
    TestTransformType::Pointer transform)
{
    using ResampleType = itk::ResampleImageFilter<TestImageType, TestImageType>;
    using InterpolatorType = itk::LinearInterpolateImageFunction<TestImageType, double>;

    auto resampler = ResampleType::New();
    resampler->SetInput(img);
    resampler->SetTransform(transform);

    auto interp = InterpolatorType::New();
    resampler->SetInterpolator(interp);

    resampler->SetSize(img->GetLargestPossibleRegion().GetSize());
    resampler->SetOutputSpacing(img->GetSpacing());
    resampler->SetOutputOrigin(img->GetOrigin());
    resampler->SetOutputDirection(img->GetDirection());
    resampler->SetDefaultPixelValue(0);
    resampler->Update();

    return resampler->GetOutput();
}


// Compute min/max of nonzero voxels in an image.
// Returns {min, max}.
inline std::pair<float, float> compute_image_range(TestImageType::Pointer img)
{
    float mn = std::numeric_limits<float>::max();
    float mx = std::numeric_limits<float>::lowest();

    using IterType = itk::ImageRegionIteratorWithIndex<TestImageType>;
    IterType it(img, img->GetLargestPossibleRegion());
    for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
        float v = it.Get();
        if (v > 1e-6) {
            mn = std::min(mn, v);
            mx = std::max(mx, v);
        }
    }
    return {mn, mx};
}


// Wrap an OkanQuadraticTransform in a CompositeTransformType for
// ConvertEddyTransformToField.
using CompositeTransformType = itk::CompositeTransform<double, 3>;

inline CompositeTransformType::Pointer wrap_in_composite(TestTransformType::Pointer transform)
{
    auto composite = CompositeTransformType::New();
    composite->AddTransform(transform);
    return composite;
}


// Simplified ChangeImageHeaderToDP for tests: sets direction to identity,
// adjusts origin to center-voxel mode (matching DIFFPREP::ChangeImageHeaderToDP
// with rot_center="center_voxel").
inline TestImageType::Pointer make_dp_image(TestImageType::Pointer img)
{
    using DupType = itk::ImageDuplicator<TestImageType>;
    auto dup = DupType::New();
    dup->SetInputImage(img);
    dup->Update();
    auto nimg = dup->GetOutput();

    TestImageType::DirectionType id_dir;
    id_dir.SetIdentity();
    nimg->SetDirection(id_dir);

    auto sz = img->GetLargestPossibleRegion().GetSize();
    auto sp = img->GetSpacing();
    TestImageType::PointType new_orig;
    new_orig[0] = -((int)(sz[0]) - 1) / 2.0 * sp[0];
    new_orig[1] = -((int)(sz[1]) - 1) / 2.0 * sp[1];
    new_orig[2] = -((int)(sz[2]) - 1) / 2.0 * sp[2];
    nimg->SetOrigin(new_orig);

    return nimg;
}


// Create a 3D mask image (all ones in interior, zero at boundaries).
inline TestImageType::Pointer create_block_mask(int sx, int sy, int sz,
                                                  double spx, double spy, double spz,
                                                  int border = 1)
{
    auto mask = create_test_image(sx, sy, sz, spx, spy, spz, 0.0f);
    itk::ImageRegionIteratorWithIndex<TestImageType> it(mask, mask->GetLargestPossibleRegion());
    for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
        auto idx = it.GetIndex();
        if (idx[0] >= border && idx[0] < sx - border &&
            idx[1] >= border && idx[1] < sy - border &&
            idx[2] >= border && idx[2] < sz - border) {
            it.Set(1.0f);
        }
    }
    return mask;
}

#endif
