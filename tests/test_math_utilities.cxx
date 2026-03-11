// Test pure math utility functions used in outlier detection and residual
// probability computation.

#include "test_macros.h"
#include "math_utilities.h"
#include <cmath>
#include <vector>


// ============================================================
// normalCDF_val tests
// ============================================================
void test_normalCDF_zero()
{
    ASSERT_NEAR(normalCDF_val(0.0), 0.5, 1e-10);
}

void test_normalCDF_positive()
{
    // CDF(1.96) ~ 0.975
    ASSERT_NEAR(normalCDF_val(1.96), 0.97500210485, 1e-4);
}

void test_normalCDF_negative()
{
    // CDF(-1.96) ~ 0.025
    ASSERT_NEAR(normalCDF_val(-1.96), 0.02499789515, 1e-4);
}

void test_normalCDF_symmetry()
{
    // CDF(x) + CDF(-x) = 1
    for (double x = 0.1; x < 4.0; x += 0.5) {
        ASSERT_NEAR(normalCDF_val(x) + normalCDF_val(-x), 1.0, 1e-10);
    }
}

void test_normalCDF_large_negative()
{
    // Implementation clamps for value < -5 (strict less-than)
    ASSERT_NEAR(normalCDF_val(-5.01), 0.0, 1e-15);
    ASSERT_NEAR(normalCDF_val(-10.0), 0.0, 1e-15);
    // -5.0 itself is NOT clamped; it goes through erfc
    ASSERT_TRUE(normalCDF_val(-5.0) > 0.0);
    ASSERT_TRUE(normalCDF_val(-5.0) < 1e-5);
}

void test_normalCDF_nan()
{
    ASSERT_NEAR(normalCDF_val(std::nan("")), 0.0, 1e-15);
}

void test_normalCDF_inf()
{
    ASSERT_NEAR(normalCDF_val(std::numeric_limits<double>::infinity()), 0.0, 1e-15);
    ASSERT_NEAR(normalCDF_val(-std::numeric_limits<double>::infinity()), 0.0, 1e-15);
}


// ============================================================
// ComputeResidProb tests
// ============================================================
void test_residprob_agg0()
{
    // norm_x = (val - mu) / sigma = (2.0 - 1.0) / 1.0 = 1.0
    // After agg0: (1.0 - 1.4) / 0.6 = -0.6667
    // 1 - CDF(-0.6667) = 1 - 0.2525 = 0.7475
    double prob = ComputeResidProb(2.0, 1.0, 1.0, 0);
    ASSERT_NEAR(prob, 1.0 - normalCDF_val((1.0 - 1.4) / 0.6), 1e-6);
}

void test_residprob_agg1()
{
    double val = 2.0, mu = 1.0, sigma = 1.0;
    double norm_x = (val - mu) / sigma;
    norm_x = (norm_x - 1.3) / 0.45;
    double expected = 1.0 - normalCDF_val(norm_x);
    ASSERT_NEAR(ComputeResidProb(2.0, 1.0, 1.0, 1), expected, 1e-6);
}

void test_residprob_agg2()
{
    double val = 2.0, mu = 1.0, sigma = 1.0;
    double norm_x = (val - mu) / sigma;
    norm_x = (norm_x - 1.2) / 0.35;
    double expected = 1.0 - normalCDF_val(norm_x);
    ASSERT_NEAR(ComputeResidProb(2.0, 1.0, 1.0, 2), expected, 1e-6);
}


// ============================================================
// log_gaussian tests
// ============================================================
void test_log_gaussian_standard()
{
    // log N(0; 0, 1) = -0.5 * log(2*pi) = -0.9189...
    EigenVecType x(1); x(0) = 0.0;
    EigenVecType result = log_gaussian(x, 0.0, 1.0);
    ASSERT_NEAR(result(0), -0.5 * std::log(2.0 * M_PI), 1e-6);
}

void test_log_gaussian_nonzero()
{
    // log N(1; 0, 1) = -0.5 * (1 + log(2*pi))
    EigenVecType x(1); x(0) = 1.0;
    EigenVecType result = log_gaussian(x, 0.0, 1.0);
    ASSERT_NEAR(result(0), -0.5 * (1.0 + std::log(2.0 * M_PI)), 1e-6);
}

void test_log_gaussian_exp_matches_gaussian()
{
    // exp(log_gaussian) should match the standard Gaussian formula
    EigenVecType x(1); x(0) = 1.5;
    float mu = 2.0, sigma = 0.5;
    EigenVecType result = log_gaussian(x, mu, sigma);
    double expected = (1.0 / (sigma * std::sqrt(2.0 * M_PI)))
                    * std::exp(-0.5 * std::pow((1.5 - mu) / sigma, 2));
    ASSERT_NEAR(std::exp(result(0)), expected, 1e-6);
}


// ============================================================
// median tests
// ============================================================
void test_median_odd()
{
    std::vector<float> v = {3, 1, 2};
    ASSERT_NEAR(median(v), 2.0, 1e-6);
}

void test_median_single()
{
    std::vector<float> v = {5};
    ASSERT_NEAR(median(v), 5.0, 1e-6);
}

void test_median_sorted()
{
    std::vector<float> v = {1, 2, 3, 4, 5};
    ASSERT_NEAR(median(v), 3.0, 1e-6);
}


// ============================================================
// round50 tests
// ============================================================
void test_round50_exact()
{
    ASSERT_EQ(round50(100), 100);
    ASSERT_EQ(round50(0), 0);
    ASSERT_EQ(round50(50), 50);
}

void test_round50_rounding()
{
    ASSERT_EQ(round50(74), 50);
    ASSERT_EQ(round50(76), 100);
    ASSERT_EQ(round50(125), 100);
    ASSERT_EQ(round50(126), 150);
}


// ============================================================
// average tests
// ============================================================
void test_average_uniform_weights()
{
    EigenVecType x(3); x << 2.0, 4.0, 6.0;
    EigenVecType w(3); w << 1.0, 1.0, 1.0;
    ASSERT_NEAR(average(x, w), 4.0, 1e-6);
}

void test_average_weighted()
{
    EigenVecType x(2); x << 0.0, 10.0;
    EigenVecType w(2); w << 3.0, 1.0;
    // weighted avg = (0*3 + 10*1) / (3+1) = 2.5
    ASSERT_NEAR(average(x, w), 2.5, 1e-6);
}


int main()
{
    std::cout << "=== Math Utilities Tests ===" << std::endl;

    TEST(normalCDF_zero);
    TEST(normalCDF_positive);
    TEST(normalCDF_negative);
    TEST(normalCDF_symmetry);
    TEST(normalCDF_large_negative);
    TEST(normalCDF_nan);
    TEST(normalCDF_inf);
    TEST(residprob_agg0);
    TEST(residprob_agg1);
    TEST(residprob_agg2);
    TEST(log_gaussian_standard);
    TEST(log_gaussian_nonzero);
    TEST(log_gaussian_exp_matches_gaussian);
    TEST(median_odd);
    TEST(median_single);
    TEST(median_sorted);
    TEST(round50_exact);
    TEST(round50_rounding);
    TEST(average_uniform_weights);
    TEST(average_weighted);

    TEST_SUMMARY();
}
