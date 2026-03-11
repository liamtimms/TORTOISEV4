// Test OkanQuadraticTransform Jacobian computations against finite differences.
// This test catches two confirmed bugs:
//   1) Line 538: m_Parameters[12] used where [13] should be (ComputeJacobianWithRespectToParameters)
//   2) Line 624: x1*x1*y1*y1 instead of x1*x1-y1*y1 (ComputeJacobianWithRespectToPosition, cubic path)

#include "test_macros.h"
#include "itkOkanQuadraticTransform.h"
#include <cmath>
#include <vector>

using TransformType = itk::OkanQuadraticTransform<double, 3, 3>;
using PointType = TransformType::InputPointType;
using ParametersType = TransformType::ParametersType;
using JacobianType = TransformType::JacobianType;

static const double FD_DELTA = 1e-5;
static const double JAC_TOL  = 1e-4;


// --- Helper: create transform with given phase and parameters ---
static TransformType::Pointer make_transform(const std::string& phase,
                                              const ParametersType& params,
                                              bool enable_cubic = false)
{
    auto t = TransformType::New();
    t->SetPhase(phase);
    t->SetIdentity();

    if (enable_cubic) {
        ParametersType flags(24);
        flags.Fill(0);
        for (int i = 14; i <= 20; i++) flags[i] = 1;
        t->SetParametersForOptimizationFlags(flags);
    }

    t->SetParameters(params);
    return t;
}

// --- Helper: build identity-like params for a phase ---
static ParametersType identity_params(int phase_idx)
{
    ParametersType p(24);
    p.Fill(0);
    p[6 + phase_idx] = 1.0;
    return p;
}


// ============================================================
// Finite-difference Jacobian w.r.t. parameters
// ============================================================
static void fd_jacobian_params(TransformType::Pointer t, const PointType& pt,
                                JacobianType& fd_jac)
{
    fd_jac.SetSize(3, 24);
    fd_jac.Fill(0);

    ParametersType p0 = t->GetParameters();

    for (int j = 0; j < 24; j++) {
        ParametersType pp = p0, pm = p0;
        pp[j] += FD_DELTA;
        pm[j] -= FD_DELTA;

        t->SetParameters(pp);
        PointType outp = t->TransformPoint(pt);
        t->SetParameters(pm);
        PointType outm = t->TransformPoint(pt);

        for (int i = 0; i < 3; i++)
            fd_jac[i][j] = (outp[i] - outm[i]) / (2.0 * FD_DELTA);
    }

    // Restore original params
    t->SetParameters(p0);
}


// ============================================================
// Finite-difference Jacobian w.r.t. position
// ============================================================
static void fd_jacobian_position(TransformType::Pointer t, const PointType& pt,
                                  JacobianType& fd_jac)
{
    fd_jac.SetSize(3, 3);
    fd_jac.Fill(0);

    for (int j = 0; j < 3; j++) {
        PointType pp = pt, pm = pt;
        pp[j] += FD_DELTA;
        pm[j] -= FD_DELTA;

        PointType outp = t->TransformPoint(pp);
        PointType outm = t->TransformPoint(pm);

        for (int i = 0; i < 3; i++)
            fd_jac[i][j] = (outp[i] - outm[i]) / (2.0 * FD_DELTA);
    }
}


// --- Helper: compare analytical vs FD jacobian ---
// skip_cols: column indices to skip (e.g., center offset params 21-23 whose
// Jacobian is intentionally not implemented since they're not optimized)
static void compare_jacobians(const JacobianType& analytical, const JacobianType& fd,
                               const std::string& label, double tol = JAC_TOL,
                               const std::vector<int>& skip_cols = {})
{
    for (unsigned int i = 0; i < analytical.rows(); i++) {
        for (unsigned int j = 0; j < analytical.cols(); j++) {
            // Skip columns that are known to be unimplemented
            bool should_skip = false;
            for (int sc : skip_cols) {
                if ((int)j == sc) { should_skip = true; break; }
            }
            if (should_skip) continue;

            double a = analytical[i][j];
            double f = fd[i][j];
            double diff = std::fabs(a - f);
            double ref = std::max(std::fabs(a), std::fabs(f));
            // Use relative tolerance for large values, absolute for small
            double effective_tol = std::max(tol, tol * ref);
            if (diff > effective_tol) {
                std::ostringstream oss;
                oss << label << ": Jacobian mismatch at [" << i << "][" << j << "]: "
                    << "analytical=" << a << ", finite_diff=" << f
                    << ", diff=" << diff << ", tol=" << effective_tol;
                throw std::runtime_error(oss.str());
            }
        }
    }
}


// ============================================================
// Test: Jacobian w.r.t. parameters at identity (vertical phase)
// ============================================================
// Helper to create a PointType
static PointType make_point(double x, double y, double z)
{
    PointType p;
    p[0] = x; p[1] = y; p[2] = z;
    return p;
}

void test_jac_params_identity_vertical()
{
    ParametersType p = identity_params(1);
    auto t = make_transform("vertical", p);

    std::vector<PointType> test_points;
    test_points.push_back(make_point(0, 0, 0));
    test_points.push_back(make_point(10, 20, 30));
    test_points.push_back(make_point(-5, 15, -10));
    test_points.push_back(make_point(100, 0, 0));

    for (auto& pt : test_points) {
        JacobianType analytical, fd;
        t->ComputeJacobianWithRespectToParameters(pt, analytical);
        fd_jacobian_params(t, pt, fd);
        // Restore params after FD perturbation
        t->SetParameters(p);
        compare_jacobians(analytical, fd, "identity_vertical", JAC_TOL, {21, 22, 23});
    }
}


// ============================================================
// Test: Jacobian w.r.t. parameters with rotation
// ============================================================
void test_jac_params_with_rotation()
{
    ParametersType p = identity_params(1);
    p[3] = 0.05; p[4] = -0.03; p[5] = 0.02;
    auto t = make_transform("vertical", p);

    PointType pt; pt[0] = 10; pt[1] = 20; pt[2] = 30;
    JacobianType analytical, fd;
    t->ComputeJacobianWithRespectToParameters(pt, analytical);
    fd_jacobian_params(t, pt, fd);
    t->SetParameters(p);
    compare_jacobians(analytical, fd, "with_rotation", JAC_TOL, {21, 22, 23});
}


// ============================================================
// Test: Jacobian w.r.t. parameters with linear eddy terms
// ============================================================
void test_jac_params_with_linear_eddy()
{
    ParametersType p = identity_params(1);
    p[6] = 0.02; p[7] = 1.03; p[8] = -0.01;
    auto t = make_transform("vertical", p);

    PointType pt; pt[0] = 15; pt[1] = -10; pt[2] = 25;
    JacobianType analytical, fd;
    t->ComputeJacobianWithRespectToParameters(pt, analytical);
    fd_jacobian_params(t, pt, fd);
    t->SetParameters(p);
    compare_jacobians(analytical, fd, "linear_eddy", JAC_TOL, {21, 22, 23});
}


// ============================================================
// CRITICAL TEST: Jacobian w.r.t. parameters with BOTH quadratic
// eddy terms active (params[12] AND params[13] nonzero).
// This catches the bug on line 538 where m_Parameters[12] is
// used instead of m_Parameters[13].
// ============================================================
void test_jac_params_both_quadratic_terms()
{
    for (int phase_idx = 0; phase_idx < 3; phase_idx++) {
        std::string phase_name = (phase_idx == 0) ? "horizontal" :
                                  (phase_idx == 1) ? "vertical" : "slice";

        ParametersType p = identity_params(phase_idx);
        // Set both quadratic terms to different nonzero values
        p[12] = 0.0005;
        p[13] = 0.0003;
        // Also add some rotation to make the chain rule nontrivial
        p[3] = 0.02; p[4] = -0.01; p[5] = 0.015;
        auto t = make_transform(phase_name, p);

        std::vector<PointType> test_points;
        test_points.push_back(make_point(10, 20, 30));
        test_points.push_back(make_point(-5, 15, -10));
        test_points.push_back(make_point(20, -15, 25));

        for (auto& pt : test_points) {
            JacobianType analytical, fd;
            t->ComputeJacobianWithRespectToParameters(pt, analytical);
            fd_jacobian_params(t, pt, fd);
            t->SetParameters(p);
            compare_jacobians(analytical, fd,
                "both_quadratic_" + phase_name, JAC_TOL, {21, 22, 23});
        }
    }
}


// ============================================================
// Test: Jacobian w.r.t. parameters with only param[12] nonzero
// (this should pass even with the bug, since the bug only
// manifests when param[13] is nonzero)
// ============================================================
void test_jac_params_single_quadratic_12()
{
    ParametersType p = identity_params(1);
    p[12] = 0.001;
    p[3] = 0.03;
    auto t = make_transform("vertical", p);

    PointType pt; pt[0] = 10; pt[1] = 20; pt[2] = 5;
    JacobianType analytical, fd;
    t->ComputeJacobianWithRespectToParameters(pt, analytical);
    fd_jacobian_params(t, pt, fd);
    t->SetParameters(p);
    compare_jacobians(analytical, fd, "single_quad_12", JAC_TOL, {21, 22, 23});
}


// ============================================================
// Test: Jacobian w.r.t. parameters with only param[13] nonzero
// (this will fail because the code uses m_Parameters[12] for
// the param[13] contribution)
// ============================================================
void test_jac_params_single_quadratic_13()
{
    ParametersType p = identity_params(1);
    p[13] = 0.001;
    p[3] = 0.03;
    auto t = make_transform("vertical", p);

    PointType pt; pt[0] = 10; pt[1] = 20; pt[2] = 15;
    JacobianType analytical, fd;
    t->ComputeJacobianWithRespectToParameters(pt, analytical);
    fd_jacobian_params(t, pt, fd);
    t->SetParameters(p);
    compare_jacobians(analytical, fd, "single_quad_13", JAC_TOL, {21, 22, 23});
}


// ============================================================
// Test: Jacobian w.r.t. position at identity
// ============================================================
void test_jac_position_identity()
{
    ParametersType p = identity_params(1);
    auto t = make_transform("vertical", p);

    PointType pt; pt[0] = 10; pt[1] = 20; pt[2] = 30;
    JacobianType analytical, fd;
    t->ComputeJacobianWithRespectToPosition(pt, analytical);
    fd_jacobian_position(t, pt, fd);
    compare_jacobians(analytical, fd, "position_identity");
}


// ============================================================
// Test: Jacobian w.r.t. position with eddy terms
// ============================================================
void test_jac_position_with_eddy()
{
    for (int phase_idx = 0; phase_idx < 3; phase_idx++) {
        std::string phase_name = (phase_idx == 0) ? "horizontal" :
                                  (phase_idx == 1) ? "vertical" : "slice";

        ParametersType p = identity_params(phase_idx);
        p[3] = 0.01; p[4] = -0.02;
        p[9] = 0.0001; p[10] = 0.00015; p[11] = -0.0001;
        p[12] = 0.0003; p[13] = 0.0002;
        auto t = make_transform(phase_name, p);

        PointType pt; pt[0] = 10; pt[1] = 20; pt[2] = 15;
        JacobianType analytical, fd;
        t->ComputeJacobianWithRespectToPosition(pt, analytical);
        fd_jacobian_position(t, pt, fd);
        compare_jacobians(analytical, fd, "position_eddy_" + phase_name);
    }
}


// ============================================================
// CRITICAL TEST: Jacobian w.r.t. position in cubic path.
// This catches the bug on line 624 where x1*x1*y1*y1 (multiply)
// is used instead of x1*x1-y1*y1 (subtract).
// ============================================================
void test_jac_position_cubic()
{
    for (int phase_idx = 0; phase_idx < 3; phase_idx++) {
        std::string phase_name = (phase_idx == 0) ? "horizontal" :
                                  (phase_idx == 1) ? "vertical" : "slice";

        ParametersType p = identity_params(phase_idx);
        p[3] = 0.01;
        p[12] = 0.0003; p[13] = 0.0002;
        // Set a cubic term to trigger the cubic code path
        p[14] = 0.00001;
        auto t = make_transform(phase_name, p, true);

        std::vector<PointType> test_points;
        test_points.push_back(make_point(10, 20, 15));
        test_points.push_back(make_point(-5, 8, 12));
        test_points.push_back(make_point(15, -10, 20));

        for (auto& pt : test_points) {
            JacobianType analytical, fd;
            t->ComputeJacobianWithRespectToPosition(pt, analytical);
            fd_jacobian_position(t, pt, fd);
            compare_jacobians(analytical, fd,
                "position_cubic_" + phase_name);
        }
    }
}


// ============================================================
// Test: Jacobian determinant sanity checks
// ============================================================
void test_jac_determinant_sanity()
{
    // Near-identity: determinant should be close to 1
    ParametersType p = identity_params(1);
    p[3] = 0.001; // tiny rotation
    auto t = make_transform("vertical", p);

    PointType pt; pt[0] = 10; pt[1] = 20; pt[2] = 30;
    JacobianType jac;
    t->ComputeJacobianWithRespectToPosition(pt, jac);

    // 3x3 determinant
    double det = jac[0][0] * (jac[1][1]*jac[2][2] - jac[1][2]*jac[2][1])
               - jac[0][1] * (jac[1][0]*jac[2][2] - jac[1][2]*jac[2][0])
               + jac[0][2] * (jac[1][0]*jac[2][1] - jac[1][1]*jac[2][0]);
    ASSERT_NEAR(det, 1.0, 0.01);

    // With moderate eddy: determinant should still be positive
    p[12] = 0.001; p[13] = 0.0005;
    t->SetParameters(p);
    t->ComputeJacobianWithRespectToPosition(pt, jac);
    det = jac[0][0] * (jac[1][1]*jac[2][2] - jac[1][2]*jac[2][1])
        - jac[0][1] * (jac[1][0]*jac[2][2] - jac[1][2]*jac[2][0])
        + jac[0][2] * (jac[1][0]*jac[2][1] - jac[1][1]*jac[2][0]);
    ASSERT_TRUE(det > 0);
}


int main()
{
    std::cout << "=== OkanQuadraticTransform Jacobian Tests ===" << std::endl;

    TEST(jac_params_identity_vertical);
    TEST(jac_params_with_rotation);
    TEST(jac_params_with_linear_eddy);
    TEST(jac_params_both_quadratic_terms);
    TEST(jac_params_single_quadratic_12);
    TEST(jac_params_single_quadratic_13);
    TEST(jac_position_identity);
    TEST(jac_position_with_eddy);
    TEST(jac_position_cubic);
    TEST(jac_determinant_sanity);

    TEST_SUMMARY();
}
