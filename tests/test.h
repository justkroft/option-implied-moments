#ifndef TEST_H
#define TEST_H

#define PASS "PASS"
#define FAIL "FAIL"

#include <stdio.h>
#include <string.h>
#include <math.h>

extern int _tests_run;
extern int _tests_failed;


int assertTrue(int condition, const char* label);
int assertNear(double actual, double expected, double tol, const char* label);

#endif
