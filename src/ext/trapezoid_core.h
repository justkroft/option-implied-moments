#ifndef TRAPEZOID_CORE_H
#define TRAPEZOID_CORE_H

/*
* C implementation of the BKM trapezoidal method for computing risk-neutral moments
* from option prices.
*/

#include <stddef.h>

/* Return codes */
#define TRAPZ_OK 0             // success
#define TRAPZ_ERR_ALLOC -1     // malloc failed
#define TRAPZ_ERR_NO_CALLS -2  // no OTM call options provided
#define TRAPZ_ERR_NO_PUTS -3   // no OTM put options provided
#define TRAPZ_ERR_FEW_OPT -4   // too few options to perform trapezoidal integration (less than 4)


/* Option flag encoding */
#define OPT_CALL 1
#define OPT_PUT 0

typedef struct {
    double var;
    double skew;
    double kurt;
} TrapezoidResult;


/* ----
*
* Compute risk-neutral variance, skewness, and kurtosis from option prices using the BKM
* trapezoidal method for a single stock-month group.
*
* Parameters
* ----------
* n          : number of OTM options in a stock-month group
* strikes    : OTM strike prices,  length n
* ivols      : implied volatilities, length n (parallel to strikes)
* flags      : OPT_CALL / OPT_PUT,  length n (parallel to strikes)
* spot       : spot price S of the underlying
* r          : continuously-compounded risk-free rate
* T          : time-to-maturity in years
* out        : pointer to result struct; set to NaN on error paths
*
* ---- */
int trapz_moments(
    size_t n,
    const double *strikes,
    const double *ivols,
    const int *flags,
    double spot,
    double r,
    double T,
    TrapezoidResult *out
);

#endif
