// Tests that the 8 new S2V command-line flags are correctly parsed by
// TORTOISE_PARSER and return expected values from their getters.

#include "test_macros.h"
#include "TORTOISE_parser.h"
#include <cstring>


// --- Helper: construct a TORTOISE_PARSER from a vector of C strings ---
static TORTOISE_PARSER* make_parser(std::vector<std::string> args)
{
    // Build argc/argv from the string vector
    std::vector<char*> argv;
    for (auto& s : args)
        argv.push_back(const_cast<char*>(s.c_str()));

    return new TORTOISE_PARSER(static_cast<int>(argv.size()), argv.data());
}


// ============================================================
// Test: --s2v_warm_start 1 → getS2VWarmStart() returns true
// ============================================================
void test_cli_s2v_warm_start()
{
    auto parser = make_parser({"test", "--up_data", "/tmp/fake.nii", "--s2v_warm_start", "1"});
    ASSERT_EQ((int)parser->getS2VWarmStart(), 1);
    delete parser;
}


// ============================================================
// Test: --s2v_convergence_threshold 0.005 → getter returns 0.005
// ============================================================
void test_cli_s2v_convergence_threshold()
{
    auto parser = make_parser({"test", "--up_data", "/tmp/fake.nii", "--s2v_convergence_threshold", "0.005"});
    ASSERT_NEAR(parser->getS2VConvergenceThreshold(), 0.005f, 1e-6);
    delete parser;
}


// ============================================================
// Test: --large_motion_correction 1 → getter returns true
// ============================================================
void test_cli_large_motion_correction()
{
    auto parser = make_parser({"test", "--up_data", "/tmp/fake.nii", "--large_motion_correction", "1"});
    ASSERT_EQ((int)parser->getLargeMotionCorrection(), 1);
    delete parser;
}


// ============================================================
// Test: --s2v_multistart 1 → getter returns true
// ============================================================
void test_cli_s2v_multistart()
{
    auto parser = make_parser({"test", "--up_data", "/tmp/fake.nii", "--s2v_multistart", "1"});
    ASSERT_EQ((int)parser->getS2VMultistart(), 1);
    delete parser;
}


// ============================================================
// Test: --s2v_smoothing_schedule 1.0,0.5,0.0 → getter returns string
// ============================================================
void test_cli_s2v_smoothing_schedule()
{
    auto parser = make_parser({"test", "--up_data", "/tmp/fake.nii", "--s2v_smoothing_schedule", "1.0,0.5,0.0"});
    ASSERT_EQ(parser->getS2VSmoothingSchedule(), std::string("1.0,0.5,0.0"));
    delete parser;
}


// ============================================================
// Test: --mapmri_degree_schedule 2,4,4 → getter returns string
// ============================================================
void test_cli_mapmri_degree_schedule()
{
    auto parser = make_parser({"test", "--up_data", "/tmp/fake.nii", "--mapmri_degree_schedule", "2,4,4"});
    ASSERT_EQ(parser->getMAPMRIDegreeSchedule(), std::string("2,4,4"));
    delete parser;
}


// ============================================================
// Test: --s2v_lambda 0.5 → getter returns 0.5
// ============================================================
void test_cli_s2v_lambda()
{
    auto parser = make_parser({"test", "--up_data", "/tmp/fake.nii", "--s2v_lambda", "0.5"});
    ASSERT_NEAR(parser->getS2VLambda(), 0.5f, 1e-6);
    delete parser;
}


// ============================================================
// Test: --s2v_niter 5 → getter returns 5
// ============================================================
void test_cli_s2v_niter()
{
    auto parser = make_parser({"test", "--up_data", "/tmp/fake.nii", "--s2v_niter", "5"});
    ASSERT_EQ(parser->getS2VNiter(), 5);
    delete parser;
}


// ============================================================
// Test: no new flags → all 8 getters return defaults
// ============================================================
void test_cli_defaults_when_absent()
{
    auto parser = make_parser({"test", "--up_data", "/tmp/fake.nii"});
    ASSERT_EQ((int)parser->getS2VWarmStart(), 0);
    ASSERT_NEAR(parser->getS2VConvergenceThreshold(), 0.0f, 1e-10);
    ASSERT_EQ((int)parser->getLargeMotionCorrection(), 0);
    ASSERT_EQ((int)parser->getS2VMultistart(), 0);
    ASSERT_EQ(parser->getS2VSmoothingSchedule(), std::string("0"));
    ASSERT_EQ(parser->getMAPMRIDegreeSchedule(), std::string("4"));
    ASSERT_NEAR(parser->getS2VLambda(), 0.0f, 1e-10);
    ASSERT_EQ(parser->getS2VNiter(), 1);
    delete parser;
}


int main()
{
    std::cout << "=== CLI Parsing Tests ===" << std::endl;

    TEST(cli_s2v_warm_start);
    TEST(cli_s2v_convergence_threshold);
    TEST(cli_large_motion_correction);
    TEST(cli_s2v_multistart);
    TEST(cli_s2v_smoothing_schedule);
    TEST(cli_mapmri_degree_schedule);
    TEST(cli_s2v_lambda);
    TEST(cli_s2v_niter);
    TEST(cli_defaults_when_absent);

    TEST_SUMMARY();
}
