#ifndef TEST_MACROS_H
#define TEST_MACROS_H

#include <iostream>
#include <cmath>
#include <string>
#include <sstream>
#include <stdexcept>

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) \
    std::cout << "  " << #name << "... "; \
    try { test_##name(); tests_passed++; std::cout << "PASS" << std::endl; } \
    catch(std::exception &e) { tests_failed++; std::cout << "FAIL: " << e.what() << std::endl; }

#define ASSERT_NEAR(a, b, tol) \
    do { \
        double _a = (a), _b = (b), _tol = (tol); \
        if(std::fabs(_a - _b) > _tol) { \
            std::ostringstream _oss; \
            _oss << "ASSERT_NEAR failed: " << #a << " = " << _a \
                 << ", " << #b << " = " << _b \
                 << ", diff = " << std::fabs(_a - _b) \
                 << ", tol = " << _tol \
                 << " [" << __FILE__ << ":" << __LINE__ << "]"; \
            throw std::runtime_error(_oss.str()); \
        } \
    } while(0)

#define ASSERT_EQ(a, b) \
    do { \
        auto _a = (a); auto _b = (b); \
        if(_a != _b) { \
            std::ostringstream _oss; \
            _oss << "ASSERT_EQ failed: " << #a << " = " << _a \
                 << ", " << #b << " = " << _b \
                 << " [" << __FILE__ << ":" << __LINE__ << "]"; \
            throw std::runtime_error(_oss.str()); \
        } \
    } while(0)

#define ASSERT_TRUE(cond) \
    do { \
        if(!(cond)) { \
            std::ostringstream _oss; \
            _oss << "ASSERT_TRUE failed: " << #cond \
                 << " [" << __FILE__ << ":" << __LINE__ << "]"; \
            throw std::runtime_error(_oss.str()); \
        } \
    } while(0)

#define TEST_SUMMARY() \
    do { \
        std::cout << std::endl << "Results: " << tests_passed << " passed, " \
                  << tests_failed << " failed" << std::endl; \
        return tests_failed > 0 ? 1 : 0; \
    } while(0)

#endif
