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


static void test_bs_price(void) {
    /* Test known values for call and put prices */
    double S = 100.0;
    double K = 100.0;
    double r = 0.05;
    double T = 1.0;
    double sigma = 0.2;

    double call_price = bs_price(S, K, r, T, sigma, OPT_CALL);
    double put_price = bs_price(S, K, r, T, sigma, OPT_PUT);

    assertNear(call_price, 10.4506, 1e-4, "Call price should be approximately 10.45058357");
    assertNear(put_price, 5.5735, 1e-4, "Put price should be approximately 5.57351735");

    /* Test put-call partity: C - S == S - K * exp(-rT) */
    struct { double S, K, r, T, sigma; } pcp_cases[] = {
        { 100,  100, 0.05, 1.00, 0.20 },
        { 100,  110, 0.05, 0.25, 0.25 },
        {  80,   75, 0.02, 0.50, 0.15 },
        { 200,  180, 0.04, 2.00, 0.30 },
        { 100,  100, 0.00, 0.50, 0.10 },   /* zero interest rate */
    };
    for (int i = 0; i < (int)(sizeof(pcp_cases)/sizeof(pcp_cases[0])); ++i) {
        double S = pcp_cases[i].S, K  = pcp_cases[i].K;
        double r = pcp_cases[i].r, T  = pcp_cases[i].T;
        double s = pcp_cases[i].sigma;
        double lhs = bs_price(S, K, r, T, s, OPT_CALL ) - bs_price(S, K, r, T, s, OPT_PUT);
        double rhs = S - K * exp(-r * T);
        char label[80];
        snprintf(
            label, sizeof(label),
            "put-call parity  S=%.0f K=%.0f r=%.2f T=%.2f sigma=%.2f",
            S, K, r, T, s
        );
        assertNear(lhs, rhs, 1e-6, label);
    }

    /* Deep OTM options approach zero */
    assertNear(
        bs_price(100, 300, 0.05, 1.0, 0.2, OPT_CALL),
        0.0, 1e-6, "Deep OTM call should be approximately 0"
    );
    assertNear(
        bs_price(100,  5, 0.05, 1.0, 0.2, OPT_PUT),
        0.0, 1e-6, "Deep OTM put should be approximately 0"
    );

    /* Zero volatility: call = max(S - K * exp(-rT), 0), put = max(K * exp(-rT) - S, 0) */
    double eps_vol = 1e-9;
    double disc = exp(-0.05 * 1.0);
    assertNear(
        bs_price(100, 90, 0.05, 1.0, eps_vol, OPT_CALL),
        fmax(100 - 90 * disc, 0), 1e-7,
        "Zero vol call should be max(S - K*exp(-rT), 0)"
    );
    assertNear(
        bs_price(100, 110, 0.05, 1.0, eps_vol, OPT_PUT),
        fmax(110 * disc - 100, 0), 1e-7,
        "Zero vol put should be max(K*exp(-rT) - S, 0)"
    );
}


int main(void) {
    test_norm_cdf();
    test_bs_price();

    printf("\n%d tests run, %d failed\n", _tests_run, _tests_failed);
    return _tests_failed == 0 ? 0 : 1;
}
