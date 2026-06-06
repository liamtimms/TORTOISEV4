// Unit test for TemporalRegularizeS2VTransforms algorithm
// Tests the temporal smoothing math using real ITK transform types.
// Build requires ITK (OkanQuadraticTransform + VNL).

#include "itkOkanQuadraticTransform.h"
#include "vnl/vnl_matrix.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using OkanQuadraticTransformType = itk::OkanQuadraticTransform<double, 3, 3>;

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) \
    std::cout << "  " << #name << "... "; \
    try { test_##name(); tests_passed++; std::cout << "PASS" << std::endl; } \
    catch(std::exception &e) { tests_failed++; std::cout << "FAIL: " << e.what() << std::endl; }

#define ASSERT_NEAR(a, b, tol) \
    if(std::fabs((a) - (b)) > (tol)) throw std::runtime_error( \
        std::string("Expected ") + std::to_string(b) + " +/- " + std::to_string(tol) + \
        " but got " + std::to_string(a) + " at line " + std::to_string(__LINE__))

// Reimplementation of the algorithm from DIFFPREP::TemporalRegularizeS2VTransforms
// so we can test it without instantiating DIFFPREP.
void TemporalRegularize(
    std::vector<OkanQuadraticTransformType::Pointer>& s2v_trans,
    vnl_matrix<int> slspec,
    float lambda)
{
    if(lambda <= 0) return;

    int Nexc = slspec.rows();
    int MB = slspec.cols();
    if(Nexc < 3) return;

    int nparams = 6;

    vnl_matrix<double> params(Nexc, nparams);
    for(int e = 0; e < Nexc; e++)
    {
        if(s2v_trans.size() <= (size_t)slspec(e,0) || !s2v_trans[slspec(e,0)])
            return;
        auto p = s2v_trans[slspec(e,0)]->GetParameters();
        for(int j = 0; j < nparams; j++)
            params(e, j) = p[j];
    }

    vnl_matrix<double> smoothed(Nexc, nparams);
    for(int j = 0; j < nparams; j++)
    {
        smoothed(0, j) = (2.0*params(0,j) + params(1,j)) / 3.0;
        for(int e = 1; e < Nexc-1; e++)
            smoothed(e, j) = (params(e-1,j) + 2.0*params(e,j) + params(e+1,j)) / 4.0;
        smoothed(Nexc-1, j) = (params(Nexc-2,j) + 2.0*params(Nexc-1,j)) / 3.0;
    }

    for(int e = 0; e < Nexc; e++)
    {
        auto p = s2v_trans[slspec(e,0)]->GetParameters();
        auto new_p = p;
        for(int j = 0; j < nparams; j++)
            new_p[j] = (1.0 - lambda) * params(e,j) + lambda * smoothed(e,j);

        for(int kk = 0; kk < MB; kk++)
        {
            if((size_t)slspec(e,kk) < s2v_trans.size() && s2v_trans[slspec(e,kk)])
                s2v_trans[slspec(e,kk)]->SetParameters(new_p);
        }
    }
}

// Helper: create a transform with given rigid params (first 6 of 24)
OkanQuadraticTransformType::Pointer make_transform(double tx, double ty, double tz,
                                                    double rx, double ry, double rz,
                                                    std::string phase = "horizontal")
{
    auto t = OkanQuadraticTransformType::New();
    t->SetPhase(phase);
    t->SetIdentity();
    auto p = t->GetParameters();
    p[0] = tx; p[1] = ty; p[2] = tz;
    p[3] = rx; p[4] = ry; p[5] = rz;
    t->SetParameters(p);
    return t;
}

// Helper: build slspec for Nexc excitation groups with given MB
// Simple sequential layout: group e has slices [e*MB, e*MB+1, ..., e*MB+MB-1]
vnl_matrix<int> make_slspec(int Nexc, int MB)
{
    vnl_matrix<int> slspec(Nexc, MB);
    for(int e = 0; e < Nexc; e++)
        for(int m = 0; m < MB; m++)
            slspec(e, m) = e * MB + m;
    return slspec;
}

// --- Tests ---
// TemporalRegularize smooths S2V rigid parameters across excitation groups using
// a [1,2,1]/4 moving-average kernel (interior) and [2,1]/3 at boundaries. This
// suppresses high-frequency noise in per-slice transforms while preserving
// genuine linear motion trends. Lambda blends between original (0) and smoothed
// (1). The algorithm operates on excitation-group order, not slice order, because
// that's the temporal acquisition sequence.

void test_lambda_zero_no_change()
{
    // Lambda=0 should leave transforms unchanged
    int Nexc = 5, MB = 1;
    auto slspec = make_slspec(Nexc, MB);
    int Nslices = Nexc * MB;

    std::vector<OkanQuadraticTransformType::Pointer> trans(Nslices);
    for(int k = 0; k < Nslices; k++)
        trans[k] = make_transform(k * 1.0, 0, 0, 0, 0, 0);

    TemporalRegularize(trans, slspec, 0.0f);

    for(int k = 0; k < Nslices; k++)
        ASSERT_NEAR(trans[k]->GetParameters()[0], k * 1.0, 1e-10);
}

void test_fewer_than_3_groups_no_change()
{
    // With only 2 excitation groups, function should return without modifying
    int Nexc = 2, MB = 1;
    auto slspec = make_slspec(Nexc, MB);

    std::vector<OkanQuadraticTransformType::Pointer> trans(2);
    trans[0] = make_transform(10.0, 0, 0, 0, 0, 0);
    trans[1] = make_transform(20.0, 0, 0, 0, 0, 0);

    TemporalRegularize(trans, slspec, 1.0f);

    ASSERT_NEAR(trans[0]->GetParameters()[0], 10.0, 1e-10);
    ASSERT_NEAR(trans[1]->GetParameters()[0], 20.0, 1e-10);
}

void test_constant_input_unchanged()
{
    // If all transforms have the same params, smoothing should not change them
    int Nexc = 5, MB = 1;
    auto slspec = make_slspec(Nexc, MB);

    std::vector<OkanQuadraticTransformType::Pointer> trans(5);
    for(int k = 0; k < 5; k++)
        trans[k] = make_transform(3.0, 2.0, 1.0, 0.1, 0.2, 0.3);

    TemporalRegularize(trans, slspec, 1.0f);

    for(int k = 0; k < 5; k++)
    {
        auto p = trans[k]->GetParameters();
        ASSERT_NEAR(p[0], 3.0, 1e-10);
        ASSERT_NEAR(p[1], 2.0, 1e-10);
        ASSERT_NEAR(p[2], 1.0, 1e-10);
        ASSERT_NEAR(p[3], 0.1, 1e-10);
        ASSERT_NEAR(p[4], 0.2, 1e-10);
        ASSERT_NEAR(p[5], 0.3, 1e-10);
    }
}

// Verifies the [1,2,1]/4 kernel weights: a step discontinuity (0→10 between groups
// 2 and 3) should be smoothed at the transition but preserved at the boundaries.
// This catches off-by-one errors in the kernel indexing.
void test_step_function_smoothed()
{
    // Step function: [0, 0, 0, 10, 10] in tx
    // Interior groups should move toward neighbors
    int Nexc = 5, MB = 1;
    auto slspec = make_slspec(Nexc, MB);

    std::vector<OkanQuadraticTransformType::Pointer> trans(5);
    trans[0] = make_transform(0, 0, 0, 0, 0, 0);
    trans[1] = make_transform(0, 0, 0, 0, 0, 0);
    trans[2] = make_transform(0, 0, 0, 0, 0, 0);
    trans[3] = make_transform(10, 0, 0, 0, 0, 0);
    trans[4] = make_transform(10, 0, 0, 0, 0, 0);

    TemporalRegularize(trans, slspec, 1.0f);  // Full smoothing

    // Group 2 (interior): smoothed = (0 + 2*0 + 10)/4 = 2.5
    ASSERT_NEAR(trans[2]->GetParameters()[0], 2.5, 1e-10);
    // Group 3 (interior): smoothed = (0 + 2*10 + 10)/4 = 7.5
    ASSERT_NEAR(trans[3]->GetParameters()[0], 7.5, 1e-10);
    // Group 0 (boundary): smoothed = (2*0 + 0)/3 = 0
    ASSERT_NEAR(trans[0]->GetParameters()[0], 0.0, 1e-10);
    // Group 4 (boundary): smoothed = (10 + 2*10)/3 = 10
    ASSERT_NEAR(trans[4]->GetParameters()[0], 10.0, 1e-10);
}

void test_lambda_half_blends()
{
    // Lambda=0.5: result = 0.5*original + 0.5*smoothed
    int Nexc = 3, MB = 1;
    auto slspec = make_slspec(Nexc, MB);

    std::vector<OkanQuadraticTransformType::Pointer> trans(3);
    trans[0] = make_transform(0, 0, 0, 0, 0, 0);
    trans[1] = make_transform(10, 0, 0, 0, 0, 0);
    trans[2] = make_transform(0, 0, 0, 0, 0, 0);

    // smoothed[1] = (0 + 2*10 + 0)/4 = 5.0
    // result[1] = 0.5*10 + 0.5*5 = 7.5
    TemporalRegularize(trans, slspec, 0.5f);

    ASSERT_NEAR(trans[1]->GetParameters()[0], 7.5, 1e-10);
}

// MB consistency: after smoothing, all slices in the same excitation group must
// still have identical transforms. A bug here would create discontinuities between
// simultaneously-acquired slices in the final corrected volume.
void test_mb2_consistency()
{
    // MB=2: both slices in each group should get the same params
    int Nexc = 4, MB = 2;
    auto slspec = make_slspec(Nexc, MB);
    int Nslices = Nexc * MB;

    std::vector<OkanQuadraticTransformType::Pointer> trans(Nslices);
    for(int k = 0; k < Nslices; k++)
        trans[k] = make_transform(k * 0.5, 0, 0, 0, 0, 0);

    TemporalRegularize(trans, slspec, 1.0f);

    // Each group's two slices should be identical
    for(int e = 0; e < Nexc; e++)
    {
        int s0 = slspec(e, 0);
        int s1 = slspec(e, 1);
        for(int j = 0; j < 6; j++)
            ASSERT_NEAR(trans[s0]->GetParameters()[j], trans[s1]->GetParameters()[j], 1e-10);
    }
}

// Only rigid params (0-5: translations + rotations) should be smoothed. Eddy
// current parameters (6-23) are spatially-varying and should not be temporally
// regularized — eddy fields change with gradient direction, not with time.
void test_non_rigid_params_unchanged()
{
    // Only params 0-5 should be smoothed; params 6-23 should be unchanged
    int Nexc = 4, MB = 1;
    auto slspec = make_slspec(Nexc, MB);

    std::vector<OkanQuadraticTransformType::Pointer> trans(4);
    for(int k = 0; k < 4; k++)
    {
        auto t = OkanQuadraticTransformType::New();
        t->SetPhase("horizontal");
        t->SetIdentity();
        auto p = t->GetParameters();
        p[0] = k * 1.0;  // Will be smoothed
        p[6] = k * 2.0;  // Should NOT be smoothed
        p[10] = k * 3.0; // Should NOT be smoothed
        t->SetParameters(p);
        trans[k] = t;
    }

    // Save original non-rigid params
    std::vector<double> orig_p6(4), orig_p10(4);
    for(int k = 0; k < 4; k++)
    {
        orig_p6[k] = trans[k]->GetParameters()[6];
        orig_p10[k] = trans[k]->GetParameters()[10];
    }

    TemporalRegularize(trans, slspec, 1.0f);

    // Non-rigid params should be unchanged
    for(int k = 0; k < 4; k++)
    {
        ASSERT_NEAR(trans[k]->GetParameters()[6], orig_p6[k], 1e-10);
        ASSERT_NEAR(trans[k]->GetParameters()[10], orig_p10[k], 1e-10);
    }
}

// A [1,2,1]/4 kernel exactly preserves linear trends at interior points because
// (k-1 + 2k + k+1)/4 = k. This property ensures that genuine linear motion
// (e.g., slow drift during a long scan) is not attenuated by the regularization.
void test_linear_ramp_preserved()
{
    // A linear ramp should be nearly preserved by the moving average
    // (it's a local smoother, so boundary effects will exist, but interior is exact)
    int Nexc = 7, MB = 1;
    auto slspec = make_slspec(Nexc, MB);

    std::vector<OkanQuadraticTransformType::Pointer> trans(7);
    for(int k = 0; k < 7; k++)
        trans[k] = make_transform(k * 2.0, 0, 0, 0, 0, 0);

    TemporalRegularize(trans, slspec, 1.0f);

    // Interior points: smoothed = (prev + 2*curr + next)/4 = (2(k-1) + 2*2k + 2(k+1))/4 = 2k
    // Linear ramp is preserved exactly at interior points
    for(int k = 1; k < 6; k++)
        ASSERT_NEAR(trans[k]->GetParameters()[0], k * 2.0, 1e-10);
}

int main()
{
    std::cout << "=== Temporal Regularization Tests ===" << std::endl;

    TEST(lambda_zero_no_change);
    TEST(fewer_than_3_groups_no_change);
    TEST(constant_input_unchanged);
    TEST(step_function_smoothed);
    TEST(lambda_half_blends);
    TEST(mb2_consistency);
    TEST(non_rigid_params_unchanged);
    TEST(linear_ramp_preserved);

    std::cout << std::endl;
    std::cout << tests_passed << " passed, " << tests_failed << " failed." << std::endl;

    return tests_failed > 0 ? 1 : 0;
}
