#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "trapezoid_core.h"
#include "test.h"


static void test_norm_cdf(void) {
    /* Symmetry: N(0) == 0.5 */
    assertNear(norm_cdf(0.0), 0.5, 1e-7, "norm_cdf(0) should be 0.5");

    /* Well known quantiles */
    assertNear(
        norm_cdf(1.0), 0.841344746, 1e-7,
        "norm_cdf(+1) should be approximately 0.841344746"
    );
    assertNear(
        norm_cdf(-1.0), 0.158655254, 1e-7,
        "norm_cdf(-1) should be approximately 0.158655254"
    );
    assertNear(
        norm_cdf(2.0), 0.977249868, 1e-7,
        "norm_cdf(+2) should be approximately 0.977249868"
    );
    assertNear(
        norm_cdf(-2.0), 0.022750132, 1e-7,
        "norm_cdf(-2) should be approximately 0.022750132"
    );
    assertNear(
        norm_cdf(3.0), 0.998650102, 1e-7,
        "norm_cdf(+3) should be approximately 0.998650102"
    );
    assertNear(
        norm_cdf(-3.0), 0.001349898, 1e-7,
        "norm_cdf(-3) should be approximately 0.001349898"
    );

    /* Symmetry: N(-x) + N(x) == 1 */
    double xs[] = { -3.5, -1.23, 0.0, 0.77, 2.1, 3.9 };
    for (int i = 0; i < sizeof(xs)/sizeof(xs[0]); ++i) {
        double x = xs[i];
        double sum = norm_cdf(-x) + norm_cdf(x);
        assertNear(sum, 1.0, 1e-7, "norm_cdf(-x) + norm_cdf(x) should be 1");
    }

    /* Monotonicity: N must be non-decreasing */
    double prev = 0.0;
    int mono_ok = 1;
    for (int i = -5; i < 5; ++i) {
        double x = i * 1.0;
        double cdf = norm_cdf(x);
        if (cdf < prev) {
            mono_ok = 0;
            break;
        }
        prev = cdf;
    }
    assertTrue(mono_ok, "norm_cdf should be non-decreasing");

    /* Limiting behavior: N(x) -> 0 as x -> -inf, N(x) -> 1 as x -> inf */
    assertNear(norm_cdf(-100.0), 0.0, 1e-7, "norm_cdf(-100) should be approximately 0");
    assertNear(norm_cdf(100.0), 1.0, 1e-7, "norm_cdf(100) should be approximately 1");
}


int main(void) {
    test_norm_cdf();

    printf("\n%d tests run, %d failed\n", _tests_run, _tests_failed);
    return _tests_failed == 0 ? 0 : 1;
}
