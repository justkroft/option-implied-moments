#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "trapezoid_core.h"
#include "test.h"


static void test_norm_cdf(void) {
    assertNear(
        norm_cdf(0.0), 0.5, 1e-7,
        "norm_cdf(0) should be 0.5"
    );
    assertNear(
        norm_cdf(1.0), 0.841344746, 1e-7,
        "norm_cdf(1) should be approximately 0.841344746"
    );
    assertNear(
        norm_cdf(-1.0), 0.158655254, 1e-7,
        "norm_cdf(-1) should be approximately 0.158655254"
    );
}


int main(void) {
    test_norm_cdf();

    printf("\n%d tests run, %d failed\n", _tests_run, _tests_failed);
    return _tests_failed == 0 ? 0 : 1;
}
