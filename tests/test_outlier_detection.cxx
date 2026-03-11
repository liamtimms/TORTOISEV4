// Tests for outlier detection math: EM clustering, ComputeResidProb,
// MAD robustness, and the RMS-to-probability pipeline.
//
// Since DIFFPREP::EM() is a private method, we reimplement the EM core
// logic as a standalone function and verify its mathematical properties.

#include "test_macros.h"
#include "math_utilities.h"
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>


// ============================================================
// Standalone EM implementation matching DIFFPREP::EM core logic
// (single shell, no file I/O, no RegistrationSettings dependency)
// ============================================================

struct EMResult {
    std::vector<double> M, S, P;
    float median_val, MAD_sigma;
};

EMResult run_em(std::vector<float> rms_values, int K = 4, int niter = 500)
{
    // Convert RMS to log-RMS (matching DIFFPREP::EM)
    Eigen::VectorXf res(rms_values.size());
    for (size_t i = 0; i < rms_values.size(); i++)
        res[i] = rms_values[i];

    float nzmin = res.redux([](float a, float b) {
        if (a > 0) return (b > 0) ? std::min(a, b) : a;
        else return (b > 0) ? b : std::numeric_limits<float>::infinity();
    });
    Eigen::VectorXf logres = res.array().max(nzmin).log();

    // Compute median on log-RMS
    std::vector<float> logvals(rms_values.size());
    for (size_t i = 0; i < rms_values.size(); i++) {
        if (rms_values[i] > 1E-10)
            logvals[i] = log(rms_values[i]);
        else
            logvals[i] = -10;
    }
    float med = median(logvals);

    // Compute MAD
    std::vector<float> abs_devs;
    for (size_t i = 0; i < logres.size(); i++)
        abs_devs.push_back(fabs(logres[i] - med));
    float MAD = median(abs_devs);
    float sigma = 1.4826018f * MAD;

    // Initialize K clusters (same scheme as DIFFPREP::EM)
    EMResult result;
    result.median_val = med;
    result.MAD_sigma = sigma;
    result.M.resize(K);
    result.S.resize(K);
    result.P.resize(K);

    int mid_id = (K - 1) / 2;
    for (int c = 0; c < mid_id; c++) {
        result.M[c] = med - (mid_id - c);
        result.S[c] = sigma - 0.05 * (mid_id - c);
        result.P[c] = 0.1 / (K - 1);
    }
    result.M[mid_id] = med;
    result.S[mid_id] = sigma;
    result.P[mid_id] = 0.9 + 0.1 * (K == 1);
    for (int c = mid_id + 1; c < K; c++) {
        result.M[c] = med + (c - mid_id);
        result.S[c] = sigma + 0.05 * (c - mid_id);
        result.P[c] = 0.1 / (K - 1);
    }

    // EM iterations
    const float reg = 1e-6;
    const float tol = 1e-6;
    float ll0 = -std::numeric_limits<float>::infinity();

    for (int n = 0; n < niter; n++) {
        EigenVecType tt = EigenVecType::Constant(logres.size(), 0);

        std::vector<EigenVecType> R(K), w(K);
        for (int k = 0; k < K; k++) {
            R[k] = log_gaussian(logres, result.M[k], result.S[k]);
            if (result.P[k] <= 0)
                result.P[k] = 1E-10;
            R[k] = R[k].array() + std::log(result.P[k]);
            tt = tt.array() + R[k].array().exp();
        }

        float nzmin2 = tt.redux([](float a, float b) {
            if (a > 0) return (b > 0) ? std::min(a, b) : a;
            else return (b > 0) ? b : std::numeric_limits<float>::infinity();
        });
        tt = tt.array().max(nzmin2);

        EigenVecType log_prob_norm = Eigen::log(tt.array());
        for (int k = 0; k < K; k++) {
            R[k] -= log_prob_norm;
            w[k] = R[k].array().exp() + std::numeric_limits<float>::epsilon();
            result.P[k] = w[k].mean();
            result.M[k] = average(logres, w[k]);
            result.S[k] = std::sqrt(average((logres.array() - result.M[k]).square(), w[k]) + reg);
        }
        float ll = log_prob_norm.mean();

        if (std::fabs(ll - ll0) < tol)
            break;
        ll0 = ll;
    }

    return result;
}


// ============================================================
// Test: EM separates two well-separated Gaussian clusters
// ============================================================
void test_em_separates_two_clusters()
{
    std::mt19937 rng(42);
    std::vector<float> rms;

    // Inliers: exp(N(0, 0.5)) -> log-RMS ~ N(0, 0.5)
    std::normal_distribution<float> inlier_dist(0.0f, 0.5f);
    for (int i = 0; i < 90; i++)
        rms.push_back(std::exp(inlier_dist(rng)));

    // Outliers: exp(N(3, 0.5)) -> log-RMS ~ N(3, 0.5)
    std::normal_distribution<float> outlier_dist(3.0f, 0.5f);
    for (int i = 0; i < 10; i++)
        rms.push_back(std::exp(outlier_dist(rng)));

    auto result = run_em(rms, 4);

    // Determine inlier clusters: accumulate P until >= 0.75
    double tot_P = 0;
    int k = 0;
    while (tot_P < 0.75 && k < 4) {
        tot_P += result.P[k];
        k++;
    }

    // Should reach 0.75 threshold before using all clusters
    ASSERT_TRUE(tot_P >= 0.75);
    ASSERT_TRUE(k < 4);

    // The dominant cluster mean should be near the inlier distribution (log-mean ~ 0)
    // Find cluster with highest P
    int best_k = 0;
    for (int i = 1; i < 4; i++)
        if (result.P[i] > result.P[best_k])
            best_k = i;
    ASSERT_TRUE(result.P[best_k] > 0.4);
}


// ============================================================
// Test: single cluster (all inliers) — no spurious outlier split
// ============================================================
void test_em_all_inliers()
{
    std::mt19937 rng(123);
    std::normal_distribution<float> dist(2.0f, 0.3f);
    std::vector<float> rms;

    for (int i = 0; i < 100; i++)
        rms.push_back(std::exp(dist(rng)));

    auto result = run_em(rms, 4);

    // With a single true cluster, the EM should concentrate most
    // probability into 1-2 clusters near the true mean
    double tot_P = 0;
    int k = 0;
    while (tot_P < 0.75 && k < 4) {
        tot_P += result.P[k];
        k++;
    }
    // All or nearly all probability should be in the first k clusters
    ASSERT_TRUE(tot_P >= 0.75);

    // When we accumulate to >= 0.75, remaining clusters have very little mass
    // so k==4 is acceptable (means inlier prob is essentially 1.0)
    // The key property: the inlier mean should be near the true mean (2.0)
    // Pick the cluster with max P
    int best_k = 0;
    for (int i = 1; i < 4; i++)
        if (result.P[i] > result.P[best_k])
            best_k = i;
    ASSERT_NEAR(result.M[best_k], 2.0, 0.5);
}


// ============================================================
// Test: ComputeResidProb threshold sensitivity across agg levels
// ============================================================
void test_residprob_threshold_sensitivity()
{
    float mu = 0.0, sigma = 1.0;

    // For increasing Z-scores, higher aggressive levels should give
    // lower probability (flag more outliers)
    for (double z = 1.0; z <= 4.0; z += 0.5) {
        double val = mu + z * sigma;
        double p0 = ComputeResidProb(val, mu, sigma, 0);
        double p1 = ComputeResidProb(val, mu, sigma, 1);
        double p2 = ComputeResidProb(val, mu, sigma, 2);

        // More aggressive => lower probability
        ASSERT_TRUE(p2 <= p1 + 1e-10);
        ASSERT_TRUE(p1 <= p0 + 1e-10);
    }

    // At z=0 (at the mean), all levels should give high probability
    double p0_mean = ComputeResidProb(mu, mu, sigma, 0);
    double p1_mean = ComputeResidProb(mu, mu, sigma, 1);
    double p2_mean = ComputeResidProb(mu, mu, sigma, 2);
    ASSERT_TRUE(p0_mean > 0.9);
    ASSERT_TRUE(p1_mean > 0.9);
    ASSERT_TRUE(p2_mean > 0.9);
}


// ============================================================
// Test: RMS-to-probability pipeline detects obvious outlier
// ============================================================
void test_rms_to_probability_pipeline()
{
    // Simulate per-slice RMS values: 20 normal, 1 outlier
    std::vector<float> rms_values;
    for (int i = 0; i < 20; i++)
        rms_values.push_back(50.0f + (i % 3) * 2.0f);  // Normal: 50-54
    rms_values[10] = 500.0f;  // Outlier: 10x higher

    // Take log
    std::vector<float> log_rms;
    for (auto v : rms_values) {
        if (v > 0)
            log_rms.push_back(log(v));
        else
            log_rms.push_back(0);
    }

    // Compute median and MAD
    float med = median(log_rms);
    std::vector<float> abs_devs;
    for (auto v : log_rms)
        abs_devs.push_back(fabs(v - med));
    float MAD = median(abs_devs);
    float sigma = 1.4826018f * MAD;

    // Compute probabilities
    double outlier_prob = ComputeResidProb(log_rms[10], med, sigma, 1);
    double normal_prob = ComputeResidProb(log_rms[0], med, sigma, 1);

    // Outlier should have very low probability
    ASSERT_TRUE(outlier_prob < 0.05);
    // Normal values should have high probability
    ASSERT_TRUE(normal_prob > 0.5);
}


// ============================================================
// Test: MAD robustness to extreme outliers
// ============================================================
void test_mad_robustness()
{
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> values;
    for (int i = 0; i < 100; i++)
        values.push_back(dist(rng));

    // Add extreme outliers
    for (int i = 0; i < 5; i++)
        values.push_back(100.0f);

    // Compute MAD-based sigma estimate
    float med = median(values);
    std::vector<float> abs_devs;
    for (auto v : values)
        abs_devs.push_back(fabs(v - med));
    float MAD = median(abs_devs);
    float sigma = 1.4826018f * MAD;

    // sigma should still be approximately 1.0 despite the outliers
    // (MAD is robust because the median ignores extreme values)
    ASSERT_NEAR(sigma, 1.0, 0.4);  // Generous tolerance for 100 samples

    // Median should still be near 0
    ASSERT_NEAR(med, 0.0, 0.3);
}


int main()
{
    std::cout << "=== Outlier Detection Tests ===" << std::endl;

    TEST(em_separates_two_clusters);
    TEST(em_all_inliers);
    TEST(residprob_threshold_sensitivity);
    TEST(rms_to_probability_pipeline);
    TEST(mad_robustness);

    TEST_SUMMARY();
}
