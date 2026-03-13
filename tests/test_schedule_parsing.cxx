// Unit test for parse_schedule.h
// Zero external dependencies — pure C++ standard library.
// Build: g++ -std=c++14 -I../src/main test_schedule_parsing.cxx -o test_schedule_parsing

#include "parse_schedule.h"
#include <cassert>
#include <cmath>
#include <iostream>

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) \
    std::cout << "  " << #name << "... "; \
    try { test_##name(); tests_passed++; std::cout << "PASS" << std::endl; } \
    catch(std::exception &e) { tests_failed++; std::cout << "FAIL: " << e.what() << std::endl; }

#define ASSERT_EQ(a, b) \
    if((a) != (b)) throw std::runtime_error( \
        std::string("Expected ") + std::to_string(b) + " but got " + std::to_string(a) + \
        " at line " + std::to_string(__LINE__))

#define ASSERT_FLOAT_EQ(a, b) \
    if(std::fabs((a) - (b)) > 1e-6f) throw std::runtime_error( \
        std::string("Expected ") + std::to_string(b) + " but got " + std::to_string(a) + \
        " at line " + std::to_string(__LINE__))

#define ASSERT_SIZE(vec, n) \
    if((vec).size() != (size_t)(n)) throw std::runtime_error( \
        std::string("Expected size ") + std::to_string(n) + " but got " + std::to_string((vec).size()) + \
        " at line " + std::to_string(__LINE__))

// --- parse_float_schedule tests ---

void test_float_normal()
{
    auto r = parse_float_schedule("1.0,0.5,0.0", {9.9f});
    ASSERT_SIZE(r, 3);
    ASSERT_FLOAT_EQ(r[0], 1.0f);
    ASSERT_FLOAT_EQ(r[1], 0.5f);
    ASSERT_FLOAT_EQ(r[2], 0.0f);
}

void test_float_single_value()
{
    auto r = parse_float_schedule("2.5", {9.9f});
    ASSERT_SIZE(r, 1);
    ASSERT_FLOAT_EQ(r[0], 2.5f);
}

void test_float_empty_returns_default()
{
    std::vector<float> defaults = {1.0f, 0.5f, 0.0f};
    auto r = parse_float_schedule("", defaults);
    ASSERT_SIZE(r, 3);
    ASSERT_FLOAT_EQ(r[0], 1.0f);
    ASSERT_FLOAT_EQ(r[1], 0.5f);
    ASSERT_FLOAT_EQ(r[2], 0.0f);
}

void test_float_malformed_skips_bad_tokens()
{
    // "1.0,,0.5" has an empty token in the middle — stof("") throws, so it's skipped
    auto r = parse_float_schedule("1.0,,0.5", {9.9f});
    ASSERT_SIZE(r, 2);
    ASSERT_FLOAT_EQ(r[0], 1.0f);
    ASSERT_FLOAT_EQ(r[1], 0.5f);
}

void test_float_all_bad_returns_default()
{
    auto r = parse_float_schedule("abc,def", {3.0f});
    ASSERT_SIZE(r, 1);
    ASSERT_FLOAT_EQ(r[0], 3.0f);
}

void test_float_whitespace_in_tokens()
{
    // stof handles leading whitespace
    auto r = parse_float_schedule(" 1.0, 0.5 ,0.0", {9.9f});
    ASSERT_SIZE(r, 3);
    ASSERT_FLOAT_EQ(r[0], 1.0f);
    ASSERT_FLOAT_EQ(r[1], 0.5f);
    ASSERT_FLOAT_EQ(r[2], 0.0f);
}

// --- parse_int_schedule tests ---

void test_int_normal()
{
    auto r = parse_int_schedule("2,4,4", {0});
    ASSERT_SIZE(r, 3);
    ASSERT_EQ(r[0], 2);
    ASSERT_EQ(r[1], 4);
    ASSERT_EQ(r[2], 4);
}

void test_int_single_value()
{
    auto r = parse_int_schedule("6", {0});
    ASSERT_SIZE(r, 1);
    ASSERT_EQ(r[0], 6);
}

void test_int_empty_returns_default()
{
    auto r = parse_int_schedule("", {2, 4, 4});
    ASSERT_SIZE(r, 3);
    ASSERT_EQ(r[0], 2);
    ASSERT_EQ(r[1], 4);
    ASSERT_EQ(r[2], 4);
}

void test_int_mixed_valid_invalid()
{
    auto r = parse_int_schedule("2,abc,4", {0});
    ASSERT_SIZE(r, 2);
    ASSERT_EQ(r[0], 2);
    ASSERT_EQ(r[1], 4);
}

// --- schedule_value tests ---

void test_schedule_value_in_range()
{
    std::vector<float> s = {1.0f, 0.5f, 0.0f};
    ASSERT_FLOAT_EQ(schedule_value(s, 0), 1.0f);
    ASSERT_FLOAT_EQ(schedule_value(s, 1), 0.5f);
    ASSERT_FLOAT_EQ(schedule_value(s, 2), 0.0f);
}

void test_schedule_value_beyond_range()
{
    std::vector<float> s = {1.0f, 0.5f, 0.0f};
    // Beyond schedule length: returns last element
    ASSERT_FLOAT_EQ(schedule_value(s, 3), 0.0f);
    ASSERT_FLOAT_EQ(schedule_value(s, 10), 0.0f);
    ASSERT_FLOAT_EQ(schedule_value(s, 100), 0.0f);
}

void test_schedule_value_single_element()
{
    std::vector<int> s = {4};
    ASSERT_EQ(schedule_value(s, 0), 4);
    ASSERT_EQ(schedule_value(s, 5), 4);
}

void test_schedule_value_empty()
{
    std::vector<int> s;
    // Empty schedule returns default-constructed T
    ASSERT_EQ(schedule_value(s, 0), 0);
}

int main()
{
    std::cout << "=== Schedule Parsing Tests ===" << std::endl;

    std::cout << "parse_float_schedule:" << std::endl;
    TEST(float_normal);
    TEST(float_single_value);
    TEST(float_empty_returns_default);
    TEST(float_malformed_skips_bad_tokens);
    TEST(float_all_bad_returns_default);
    TEST(float_whitespace_in_tokens);

    std::cout << "parse_int_schedule:" << std::endl;
    TEST(int_normal);
    TEST(int_single_value);
    TEST(int_empty_returns_default);
    TEST(int_mixed_valid_invalid);

    std::cout << "schedule_value:" << std::endl;
    TEST(schedule_value_in_range);
    TEST(schedule_value_beyond_range);
    TEST(schedule_value_single_element);
    TEST(schedule_value_empty);

    std::cout << std::endl;
    std::cout << tests_passed << " passed, " << tests_failed << " failed." << std::endl;

    return tests_failed > 0 ? 1 : 0;
}
