#include "test.h"

int _tests_run    = 0;
int _tests_failed = 0;

int assertTrue(int condition, const char* label) {
    _tests_run++;
    if (!(condition)) {
        fprintf(
            stderr,
            "Test failed: %s:: condition was false\n", (label)
        );
        _tests_failed++;
        return 1;
    }
    else {
        printf("Test passed: %s\n", (label));
        return 0;
    }
}


int assertNear(double actual, double expected, double tol, const char* label) {
    _tests_run++;
    if (fabs(actual - expected) > tol) {
        fprintf(
            stderr,
            "Test failed: %s:: actual %f not within %f of expected %f\n",
            (label), actual, tol, expected
        );
        _tests_failed++;
        return 1;
    }
    else {
        printf("Test passed: %s\n", (label));
        return 0;
    }
}
