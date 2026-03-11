// Test correctness of OkanQuadraticTransform::TransformPoint, ComputeMatrix,
// SetIdentity, phase handling, eddy terms, and center offset.

#include "test_macros.h"
#include "itkOkanQuadraticTransform.h"
#include <cmath>

using TransformType = itk::OkanQuadraticTransform<double, 3, 3>;
using PointType = TransformType::InputPointType;
using ParametersType = TransformType::ParametersType;

static const double PI = 3.1415926535897931;


// --- Helper: create identity transform with given phase ---
static TransformType::Pointer make_identity(const std::string& phase)
{
    auto t = TransformType::New();
    t->SetPhase(phase);
    t->SetIdentity();
    return t;
}

// --- Helper: set one parameter on an identity transform ---
static TransformType::Pointer make_with_param(const std::string& phase, int idx, double val)
{
    auto t = make_identity(phase);
    ParametersType p = t->GetParameters();
    p[idx] = val;
    t->SetParameters(p);
    return t;
}


// ============================================================
// Test: identity transform returns the input point unchanged
// ============================================================
void test_identity_horizontal()
{
    auto t = make_identity("horizontal");
    PointType in; in[0] = 10; in[1] = 20; in[2] = 30;
    PointType out = t->TransformPoint(in);
    ASSERT_NEAR(out[0], 10, 1e-12);
    ASSERT_NEAR(out[1], 20, 1e-12);
    ASSERT_NEAR(out[2], 30, 1e-12);
}

void test_identity_vertical()
{
    auto t = make_identity("vertical");
    PointType in; in[0] = -5; in[1] = 15; in[2] = -10;
    PointType out = t->TransformPoint(in);
    ASSERT_NEAR(out[0], -5, 1e-12);
    ASSERT_NEAR(out[1], 15, 1e-12);
    ASSERT_NEAR(out[2], -10, 1e-12);
}

void test_identity_slice()
{
    auto t = make_identity("slice");
    PointType in; in[0] = 100; in[1] = 0; in[2] = -50;
    PointType out = t->TransformPoint(in);
    ASSERT_NEAR(out[0], 100, 1e-12);
    ASSERT_NEAR(out[1], 0, 1e-12);
    ASSERT_NEAR(out[2], -50, 1e-12);
}


// ============================================================
// Test: identity params have correct structure
// ============================================================
void test_identity_params()
{
    // For phase=1 (vertical), identity should have params[7]=1, all others 0
    auto t = make_identity("vertical");
    ParametersType p = t->GetParameters();
    for (int i = 0; i < 24; i++) {
        if (i == 7) {
            ASSERT_NEAR(p[i], 1.0, 1e-12);
        } else {
            ASSERT_NEAR(p[i], 0.0, 1e-12);
        }
    }

    // For phase=0 (horizontal), identity should have params[6]=1
    auto t2 = make_identity("horizontal");
    ParametersType p2 = t2->GetParameters();
    ASSERT_NEAR(p2[6], 1.0, 1e-12);
    ASSERT_NEAR(p2[7], 0.0, 1e-12);

    // For phase=2 (slice), identity should have params[8]=1
    auto t3 = make_identity("slice");
    ParametersType p3 = t3->GetParameters();
    ASSERT_NEAR(p3[8], 1.0, 1e-12);
}


// ============================================================
// Test: pure translation
// ============================================================
void test_pure_translation()
{
    // With phase=vertical, translation params[0-2] shift the point,
    // but the phase coordinate (y) is then replaced by the phase polynomial.
    // For identity eddy (params[7]=1): y' = 1*(y_after_rotation + ty) = y + ty
    auto t = make_identity("vertical");
    ParametersType p = t->GetParameters();
    p[0] = 5.0;  // tx
    p[1] = -3.0; // ty
    p[2] = 2.0;  // tz
    t->SetParameters(p);

    PointType in; in[0] = 0; in[1] = 0; in[2] = 0;
    PointType out = t->TransformPoint(in);

    // After rotation (identity): p = (0,0,0) + (5,-3,2) = (5,-3,2)
    // Non-phase coords stay: x=5, z=2
    // Phase coord (y) = params[7]*(-3) = 1*(-3) = -3  (since params[6]=0, params[8]=0)
    // Wait: params[6]*p[0] + params[7]*p[1] + params[8]*p[2]
    //      = 0*5 + 1*(-3) + 0*2 = -3
    ASSERT_NEAR(out[0], 5.0, 1e-12);
    ASSERT_NEAR(out[1], -3.0, 1e-12);
    ASSERT_NEAR(out[2], 2.0, 1e-12);
}


// ============================================================
// Test: rotation matrix Rz(pi/2) maps (1,0,0) to (0,1,0)
// ============================================================
void test_rotation_z_90()
{
    auto t = make_identity("vertical");
    ParametersType p = t->GetParameters();
    p[5] = PI / 2.0;  // angleZ = 90 degrees
    t->SetParameters(p);

    // Check the rotation matrix
    auto mat = t->GetMatrix();
    // Rz(pi/2): [[0,-1,0],[1,0,0],[0,0,1]]
    ASSERT_NEAR(mat[0][0], 0.0, 1e-12);
    ASSERT_NEAR(mat[0][1], -1.0, 1e-12);
    ASSERT_NEAR(mat[1][0], 1.0, 1e-12);
    ASSERT_NEAR(mat[1][1], 0.0, 1e-12);
    ASSERT_NEAR(mat[2][2], 1.0, 1e-12);
}


// ============================================================
// Test: rotation matrix composition Rz * Ry * Rx
// ============================================================
void test_rotation_matrix_composition()
{
    double ax = PI / 6;  // 30 deg
    double ay = PI / 4;  // 45 deg
    double az = PI / 3;  // 60 deg

    auto t = make_identity("vertical");
    ParametersType p = t->GetParameters();
    p[3] = ax; p[4] = ay; p[5] = az;
    t->SetParameters(p);

    // Compute expected Rz * Ry * Rx analytically
    double cx = cos(ax), sx = sin(ax);
    double cy = cos(ay), sy = sin(ay);
    double cz = cos(az), sz = sin(az);

    // Rz * Ry * Rx
    double expected[3][3];
    expected[0][0] = cz*cy;
    expected[0][1] = cz*sy*sx - sz*cx;
    expected[0][2] = cz*sy*cx + sz*sx;
    expected[1][0] = sz*cy;
    expected[1][1] = sz*sy*sx + cz*cx;
    expected[1][2] = sz*sy*cx - cz*sx;
    expected[2][0] = -sy;
    expected[2][1] = cy*sx;
    expected[2][2] = cy*cx;

    auto mat = t->GetMatrix();
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            ASSERT_NEAR(mat[i][j], expected[i][j], 1e-12);
}


// ============================================================
// Test: phase direction controls which coordinate is replaced
// ============================================================
void test_phase_direction_horizontal()
{
    // For phase=horizontal, only x-coordinate is replaced by eddy polynomial
    auto t = make_identity("horizontal");
    ParametersType p = t->GetParameters();
    // Add a small linear eddy term in param[7] (which is the y-coefficient)
    p[7] = 0.1;
    t->SetParameters(p);

    PointType in; in[0] = 0; in[1] = 10; in[2] = 0;
    PointType out = t->TransformPoint(in);

    // After identity rotation: p = (0, 10, 0)
    // Phase=horizontal (phase=0), so x is replaced:
    //   x' = params[6]*0 + params[7]*10 + params[8]*0 = 1*0 + 0.1*10 = 1.0
    ASSERT_NEAR(out[0], 1.0, 1e-12);
    // y and z unchanged
    ASSERT_NEAR(out[1], 10.0, 1e-12);
    ASSERT_NEAR(out[2], 0.0, 1e-12);
}


// ============================================================
// Test: linear eddy scaling in phase direction
// ============================================================
void test_linear_eddy_scaling()
{
    // With phase=vertical, params[7] = 1.05 means 5% scaling in PE direction
    auto t = make_identity("vertical");
    ParametersType p = t->GetParameters();
    p[7] = 1.05;  // 5% scaling in y
    t->SetParameters(p);

    PointType in; in[0] = 0; in[1] = 100; in[2] = 0;
    PointType out = t->TransformPoint(in);

    ASSERT_NEAR(out[0], 0.0, 1e-12);
    ASSERT_NEAR(out[1], 105.0, 1e-10);
    ASSERT_NEAR(out[2], 0.0, 1e-12);
}


// ============================================================
// Test: quadratic eddy term params[12] = (x^2 - y^2)
// ============================================================
void test_quadratic_eddy_x2_y2()
{
    auto t = make_identity("vertical");
    ParametersType p = t->GetParameters();
    p[12] = 0.001;
    t->SetParameters(p);

    // Point (10, 0, 0): contribution = 0.001 * (100 - 0) = 0.1
    PointType in1; in1[0] = 10; in1[1] = 0; in1[2] = 0;
    PointType out1 = t->TransformPoint(in1);
    // y' = params[7]*0 + params[12]*(100-0) = 0 + 0.1 = 0.1
    ASSERT_NEAR(out1[1], 0.1, 1e-10);

    // Point (0, 10, 0): contribution = 0.001 * (0 - 100) = -0.1
    PointType in2; in2[0] = 0; in2[1] = 10; in2[2] = 0;
    PointType out2 = t->TransformPoint(in2);
    // y' = params[7]*10 + params[12]*(0-100) = 10 - 0.1 = 9.9
    ASSERT_NEAR(out2[1], 9.9, 1e-10);
}


// ============================================================
// Test: quadratic eddy term params[13] = (2z^2 - x^2 - y^2)
// ============================================================
void test_quadratic_eddy_2z2_x2_y2()
{
    auto t = make_identity("vertical");
    ParametersType p = t->GetParameters();
    p[13] = 0.001;
    t->SetParameters(p);

    // Point (0, 0, 10): contribution = 0.001 * (200 - 0 - 0) = 0.2
    PointType in1; in1[0] = 0; in1[1] = 0; in1[2] = 10;
    PointType out1 = t->TransformPoint(in1);
    ASSERT_NEAR(out1[1], 0.2, 1e-10);

    // Point (10, 0, 0): contribution = 0.001 * (0 - 100 - 0) = -0.1
    PointType in2; in2[0] = 10; in2[1] = 0; in2[2] = 0;
    PointType out2 = t->TransformPoint(in2);
    ASSERT_NEAR(out2[1], -0.1, 1e-10);
}


// ============================================================
// Test: center offset (params 21-23) shifts rotation center
// ============================================================
void test_center_offset()
{
    // With a center offset, rotation should happen around that center
    auto t = make_identity("vertical");
    ParametersType p = t->GetParameters();
    p[21] = 50; p[22] = 50; p[23] = 50;  // center
    t->SetParameters(p);

    // Point at the center should not move
    PointType center_pt; center_pt[0] = 50; center_pt[1] = 50; center_pt[2] = 50;
    PointType out = t->TransformPoint(center_pt);
    ASSERT_NEAR(out[0], 50, 1e-10);
    ASSERT_NEAR(out[1], 50, 1e-10);
    ASSERT_NEAR(out[2], 50, 1e-10);
}


void test_center_offset_with_rotation()
{
    // With center at (50,50,50) and small Z rotation, the center stays fixed
    auto t = make_identity("vertical");
    ParametersType p = t->GetParameters();
    p[5] = 0.01;  // small Z rotation
    p[21] = 50; p[22] = 50; p[23] = 50;
    t->SetParameters(p);

    PointType center_pt; center_pt[0] = 50; center_pt[1] = 50; center_pt[2] = 50;
    PointType out_center = t->TransformPoint(center_pt);
    // Center should approximately stay at (50, 50, 50)
    // After subtracting center: (0,0,0), rotation of (0,0,0) is (0,0,0),
    // then add center back: (50,50,50)
    ASSERT_NEAR(out_center[0], 50, 1e-10);
    ASSERT_NEAR(out_center[1], 50, 1e-10);
    ASSERT_NEAR(out_center[2], 50, 1e-10);

    // Point away from center should move
    PointType off_pt; off_pt[0] = 60; off_pt[1] = 50; off_pt[2] = 50;
    PointType out_off = t->TransformPoint(off_pt);
    // After subtracting center: (10,0,0)
    // After Rz(0.01): (10*cos(0.01), 10*sin(0.01), 0) ~ (10, 0.1, 0)
    // After adding translation: (10, 0.1, 0)
    // Phase (y) replaced by: params[7]*0.1 = 0.1 (approximately)
    // After adding center: (60, 50.1, 50)
    ASSERT_NEAR(out_off[0], 50 + 10*cos(0.01), 1e-6);
    ASSERT_NEAR(out_off[2], 50, 1e-10);
    // y is phase-replaced, so check that it moved
    ASSERT_TRUE(std::fabs(out_off[1] - 50) > 0.05);
}


// ============================================================
// Test: SetParameters / GetParameters roundtrip
// ============================================================
void test_set_get_parameters_roundtrip()
{
    auto t = make_identity("vertical");
    ParametersType p(24);
    for (int i = 0; i < 24; i++)
        p[i] = 0.01 * (i + 1);
    // Ensure the identity-like eddy term for phase
    p[7] = 1.0;
    t->SetParameters(p);

    ParametersType q = t->GetParameters();
    for (int i = 0; i < 24; i++)
        ASSERT_NEAR(q[i], p[i], 1e-15);
}


// ============================================================
// Test: cubic term params[14] = x*y*z
// ============================================================
void test_cubic_xyz_term()
{
    auto t = make_identity("vertical");
    ParametersType p = t->GetParameters();
    p[14] = 0.0001;
    t->SetParameters(p);

    // Need to enable cubics via optimization flags
    ParametersType flags(24);
    flags.Fill(0);
    flags[14] = 1;
    t->SetParametersForOptimizationFlags(flags);

    PointType in; in[0] = 10; in[1] = 10; in[2] = 10;
    PointType out = t->TransformPoint(in);

    // After identity rotation: p = (10, 10, 10)
    // Phase coord (y) quadratic: params[7]*10 = 10
    // Cubic: params[14]*10*10*10 = 0.0001*1000 = 0.1
    // y' = 10 + 0.1 = 10.1
    ASSERT_NEAR(out[1], 10.1, 1e-10);
    ASSERT_NEAR(out[0], 10, 1e-12);
    ASSERT_NEAR(out[2], 10, 1e-12);
}


int main()
{
    std::cout << "=== OkanQuadraticTransform Tests ===" << std::endl;

    TEST(identity_horizontal);
    TEST(identity_vertical);
    TEST(identity_slice);
    TEST(identity_params);
    TEST(pure_translation);
    TEST(rotation_z_90);
    TEST(rotation_matrix_composition);
    TEST(phase_direction_horizontal);
    TEST(linear_eddy_scaling);
    TEST(quadratic_eddy_x2_y2);
    TEST(quadratic_eddy_2z2_x2_y2);
    TEST(center_offset);
    TEST(center_offset_with_rotation);
    TEST(set_get_parameters_roundtrip);
    TEST(cubic_xyz_term);

    TEST_SUMMARY();
}
