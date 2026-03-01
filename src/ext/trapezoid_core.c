#include <math.h>
#include <stdlib.h>
#include <stddef.h>

#include "trapezoid_core.h"

/* ----
* normal cdf approximation
* ---- */
static double norm_cdf(double x) {
    static const double p  =  0.2316419;
    static const double b1 =  0.319381530;
    static const double b2 = -0.356563782;
    static const double b3 =  1.781477937;
    static const double b4 = -1.821255978;
    static const double b5 =  1.330274429;

    double sign = 1.0;
    if (x < 0.0) {
        x = -x;
        sign = -1.0;
    }

    double t = 1.0 / (1.0 + p * x);
    double t2 = t * t;
    double t3 = t2 * t;
    double t4 = t3 * t;
    double t5 = t4 * t;

    // standard normal PDF at x
    double pdf = (1.0 / sqrt(2.0 * M_PI)) * exp(-0.5 * x * x);
    double poly = b1 * t + b2 * t2 + b3 * t3 + b4 * t4 + b5 * t5;
    double cdf = 1.0 - pdf * poly;

    // recover sign
    if (sign < 0.0) {
        cdf = 1.0 - cdf;
    }

    return cdf;
}


/* ----
* Black-Scholes option price function
* ---- */
static double bs_price(
    double S,
    double K,
    double r,
    double T,
    double sigma,
    int flag
) {
    double sqrtT = sqrt(T);
    double d1 = (log (S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
    double d2 = d1 - sigma * sqrtT;
    double disc = exp(-r * T);

    if (flag == OPT_CALL) {
        return S * norm_cdf(d1) - K * disc * norm_cdf(d2);
    }
    else {
        return K * disc * norm_cdf(-d2) - S * norm_cdf(-d1);
    }
 }


/* ----
* BKM payoff-kernels
* ---- */
static inline double vc(double S, double K) {
    return (2.0 * (1.0 - log(K / S))) / (K * K);
}

static inline double vp(double S, double K) {
    return (2.0 * (1.0 + log(S / K))) / (K * K);
}

static inline double wc(double S, double K) {
    double lk = log(K / S);
    return (6.0 * lk - 3.0 * lk * lk) / (K * K);
}

static inline double wp(double S, double K) {
    double lk = log(S / K);
    return (6.0 * lk + 3.0 * lk * lk) / (K * K);
}

static inline double xc(double S, double K) {
    double lk = log(K / S);
    return (12.0 * lk * lk - 4.0 * lk * lk * lk) / (K * K);
}

static inline double xp(double S, double K) {
    double lk = log(S / K);
    return (12.0 * lk * lk + 4.0 * lk * lk * lk) / (K * K);
}


/* ----
* mu(t, tau)
* ---- */
static double bkm_mu(double r, double T, double V, double W, double X) {
    double exprt = exp(r * T);
    return exprt - 1.0
           - (exprt / 2.0) * V
           - (exprt / 6.0) * W
           - (exprt / 24.0) * X;
}
