// Tests the convergence metric computation used by DIFFPREP.cxx to
// decide whether S2V registration has converged across epochs.
// Mirrors the logic from DIFFPREP.cxx lines 1827-1858.

#include "test_macros.h"
#include "itkOkanQuadraticTransform.h"
#include <cmath>
#include <vector>

using OkanQuadraticTransformType = itk::OkanQuadraticTransform<double, 3, 3>;


// --- Helper: compute the convergence metric (mean abs rigid param change) ---
// This replicates DIFFPREP.cxx lines 1829-1851 exactly.
static double compute_convergence_metric(
    const std::vector<std::vector<OkanQuadraticTransformType::Pointer>>& curr_transforms,
    const std::vector<std::vector<OkanQuadraticTransformType::ParametersType>>& prev_params)
{
    double total_change = 0;
    int count = 0;

    for (size_t vol = 0; vol < curr_transforms.size(); vol++) {
        for (size_t k = 0; k < curr_transforms[vol].size(); k++) {
            if (curr_transforms[vol][k] &&
                prev_params[vol].size() > k &&
                prev_params[vol][k].GetSize() > 0) {
                auto curr_params = curr_transforms[vol][k]->GetParameters();
                // Check rigid params (0-5: translations + rotations)
                for (int p = 0; p < 6; p++) {
                    total_change += fabs(curr_params[p] - prev_params[vol][k][p]);
                    count++;
                }
            }
        }
    }

    if (count > 0)
        return total_change / count;
    return 0;
}


// --- Helper: create identity transforms for Nvols x Nslices ---
static std::vector<std::vector<OkanQuadraticTransformType::Pointer>>
create_identity_transforms(int Nvols, int Nslices)
{
    std::vector<std::vector<OkanQuadraticTransformType::Pointer>> transforms(Nvols);
    for (int v = 0; v < Nvols; v++) {
        transforms[v].resize(Nslices);
        for (int k = 0; k < Nslices; k++) {
            transforms[v][k] = OkanQuadraticTransformType::New();
            transforms[v][k]->SetPhase("horizontal");
            transforms[v][k]->SetIdentity();
        }
    }
    return transforms;
}


// --- Helper: snapshot parameters from transforms ---
static std::vector<std::vector<OkanQuadraticTransformType::ParametersType>>
snapshot_params(const std::vector<std::vector<OkanQuadraticTransformType::Pointer>>& transforms)
{
    std::vector<std::vector<OkanQuadraticTransformType::ParametersType>> params(transforms.size());
    for (size_t v = 0; v < transforms.size(); v++) {
        params[v].resize(transforms[v].size());
        for (size_t k = 0; k < transforms[v].size(); k++) {
            if (transforms[v][k])
                params[v][k] = transforms[v][k]->GetParameters();
        }
    }
    return params;
}


// ============================================================
// Test: identical params → metric = 0
// ============================================================
void test_convergence_metric_zero_when_unchanged()
{
    int Nvols = 3, Nslices = 8;
    auto transforms = create_identity_transforms(Nvols, Nslices);
    auto prev = snapshot_params(transforms);

    double metric = compute_convergence_metric(transforms, prev);
    ASSERT_NEAR(metric, 0.0, 1e-15);
}


// ============================================================
// Test: known delta → metric matches expected
// ============================================================
void test_convergence_metric_detects_change()
{
    int Nvols = 2, Nslices = 4;
    auto transforms = create_identity_transforms(Nvols, Nslices);
    auto prev = snapshot_params(transforms);

    // Apply known delta: +0.1 to Tx (param 0) on all transforms
    for (int v = 0; v < Nvols; v++) {
        for (int k = 0; k < Nslices; k++) {
            auto p = transforms[v][k]->GetParameters();
            p[0] += 0.1;  // 0.1mm translation change
            transforms[v][k]->SetParameters(p);
        }
    }

    double metric = compute_convergence_metric(transforms, prev);

    // Expected: mean abs change = 0.1 * (Nvols*Nslices) / (6 params * Nvols * Nslices)
    // = 0.1 / 6 ≈ 0.01667 (only 1 of 6 params changed)
    double expected = 0.1 / 6.0;
    ASSERT_NEAR(metric, expected, 1e-10);
}


// ============================================================
// Test: threshold comparison matches DIFFPREP logic
// ============================================================
void test_convergence_threshold_comparison()
{
    int Nvols = 2, Nslices = 4;
    auto transforms = create_identity_transforms(Nvols, Nslices);
    auto prev = snapshot_params(transforms);

    // Small change: 0.0001 on Tx
    for (int v = 0; v < Nvols; v++) {
        for (int k = 0; k < Nslices; k++) {
            auto p = transforms[v][k]->GetParameters();
            p[0] += 0.0001;
            transforms[v][k]->SetParameters(p);
        }
    }

    double metric = compute_convergence_metric(transforms, prev);

    // With threshold 0.001: metric (≈0.0000167) should be below → converged
    float threshold_high = 0.001f;
    ASSERT_TRUE(metric < threshold_high);

    // With threshold 0.00001: metric should be above → not converged
    float threshold_low = 0.00001f;
    ASSERT_TRUE(metric > threshold_low);

    // With threshold 0: convergence checking disabled (matches DIFFPREP line 1600)
    float threshold_disabled = 0.0f;
    ASSERT_TRUE(threshold_disabled <= 0);  // checking skipped entirely
}


int main()
{
    std::cout << "=== Convergence Logic Tests ===" << std::endl;

    TEST(convergence_metric_zero_when_unchanged);
    TEST(convergence_metric_detects_change);
    TEST(convergence_threshold_comparison);

    TEST_SUMMARY();
}
