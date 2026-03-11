// Tests for ParseJSONForSliceTiming and ComputeSlspec:
// verifies that JSON SliceTiming arrays and MultibandAccelerationFactor
// are correctly converted to FSL-style slspec matrices.

#include "test_macros.h"
#include "slice_timing_utils.h"


// ============================================================
// Test: sequential single-band (4 slices, sequential, MB=1)
// ============================================================
void test_sequential_single_band()
{
    json j;
    j["SliceTiming"] = {0.0, 0.05, 0.1, 0.15};

    vnl_matrix<int> slspec = ParseJSONForSliceTiming(j);

    // 4 unique times, MB=1 (only slice 0 has time=0)
    ASSERT_EQ(slspec.rows(), 4);
    ASSERT_EQ(slspec.cols(), 1);

    // Sequential order: slices 0,1,2,3 in time order
    ASSERT_EQ(slspec(0, 0), 0);
    ASSERT_EQ(slspec(1, 0), 1);
    ASSERT_EQ(slspec(2, 0), 2);
    ASSERT_EQ(slspec(3, 0), 3);
}


// ============================================================
// Test: interleaved single-band (slices 0,2 first, then 1,3)
// ============================================================
void test_interleaved_single_band()
{
    json j;
    // Interleaved: even slices acquired first
    j["SliceTiming"] = {0.0, 0.1, 0.05, 0.15};

    vnl_matrix<int> slspec = ParseJSONForSliceTiming(j);

    // 4 unique times, MB=1
    ASSERT_EQ(slspec.rows(), 4);
    ASSERT_EQ(slspec.cols(), 1);

    // Sorted unique times: 0.0, 0.05, 0.1, 0.15
    // Time 0.0 -> slice 0
    // Time 0.05 -> slice 2
    // Time 0.1 -> slice 1
    // Time 0.15 -> slice 3
    ASSERT_EQ(slspec(0, 0), 0);
    ASSERT_EQ(slspec(1, 0), 2);
    ASSERT_EQ(slspec(2, 0), 1);
    ASSERT_EQ(slspec(3, 0), 3);
}


// ============================================================
// Test: multiband factor 2 (4 slices, MB=2)
// ============================================================
void test_multiband_factor_2()
{
    json j;
    // Slices 0,2 acquired simultaneously at t=0; slices 1,3 at t=0.05
    j["SliceTiming"] = {0.0, 0.05, 0.0, 0.05};

    vnl_matrix<int> slspec = ParseJSONForSliceTiming(j);

    // 2 unique times, MB=2 (two slices have time=0)
    ASSERT_EQ(slspec.rows(), 2);
    ASSERT_EQ(slspec.cols(), 2);

    // Time 0.0 -> slices 0 and 2
    ASSERT_EQ(slspec(0, 0), 0);
    ASSERT_EQ(slspec(0, 1), 2);

    // Time 0.05 -> slices 1 and 3
    ASSERT_EQ(slspec(1, 0), 1);
    ASSERT_EQ(slspec(1, 1), 3);
}


// ============================================================
// Test: multiband factor 4 (all slices simultaneous)
// ============================================================
void test_multiband_factor_4()
{
    json j;
    j["SliceTiming"] = {0.0, 0.0, 0.0, 0.0};

    vnl_matrix<int> slspec = ParseJSONForSliceTiming(j);

    // 1 unique time, MB=4
    ASSERT_EQ(slspec.rows(), 1);
    ASSERT_EQ(slspec.cols(), 4);

    // All slices in one group
    ASSERT_EQ(slspec(0, 0), 0);
    ASSERT_EQ(slspec(0, 1), 1);
    ASSERT_EQ(slspec(0, 2), 2);
    ASSERT_EQ(slspec(0, 3), 3);
}


// ============================================================
// Test: no SliceTiming, with MultibandAccelerationFactor=2
// Uses ComputeSlspec fallback logic (from DIFFPREP::MotionAndEddy)
// ============================================================
void test_no_slice_timing_with_mb()
{
    json j;
    j["MultibandAccelerationFactor"] = 2;
    // No SliceTiming key

    int Nslices = 8;
    vnl_matrix<int> slspec = ComputeSlspec(j, Nslices);

    // Nexc = 8/2 = 4 excitations, MB = 2
    ASSERT_EQ(slspec.rows(), 4);
    ASSERT_EQ(slspec.cols(), 2);

    // Regular interleaving: k % Nexc = row, k / Nexc = col
    // k=0: r=0,c=0 -> slspec(0,0)=0
    // k=1: r=1,c=0 -> slspec(1,0)=1
    // k=2: r=2,c=0 -> slspec(2,0)=2
    // k=3: r=3,c=0 -> slspec(3,0)=3
    // k=4: r=0,c=1 -> slspec(0,1)=4
    // k=5: r=1,c=1 -> slspec(1,1)=5
    // k=6: r=2,c=1 -> slspec(2,1)=6
    // k=7: r=3,c=1 -> slspec(3,1)=7
    ASSERT_EQ(slspec(0, 0), 0);  ASSERT_EQ(slspec(0, 1), 4);
    ASSERT_EQ(slspec(1, 0), 1);  ASSERT_EQ(slspec(1, 1), 5);
    ASSERT_EQ(slspec(2, 0), 2);  ASSERT_EQ(slspec(2, 1), 6);
    ASSERT_EQ(slspec(3, 0), 3);  ASSERT_EQ(slspec(3, 1), 7);
}


// ============================================================
// Test: no SliceTiming, no MB — identity slspec
// ============================================================
void test_no_slice_timing_no_mb()
{
    json j;
    // No SliceTiming, no MultibandAccelerationFactor

    int Nslices = 6;
    vnl_matrix<int> slspec = ComputeSlspec(j, Nslices);

    // 6 rows x 1 col identity
    ASSERT_EQ(slspec.rows(), 6);
    ASSERT_EQ(slspec.cols(), 1);

    for (int k = 0; k < 6; k++)
        ASSERT_EQ(slspec(k, 0), k);
}


int main()
{
    std::cout << "=== Slice Timing Tests ===" << std::endl;

    TEST(sequential_single_band);
    TEST(interleaved_single_band);
    TEST(multiband_factor_2);
    TEST(multiband_factor_4);
    TEST(no_slice_timing_with_mb);
    TEST(no_slice_timing_no_mb);

    TEST_SUMMARY();
}
