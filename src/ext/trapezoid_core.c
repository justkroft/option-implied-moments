#include <math.h>
#include <stdlib.h>
#include <stddef.h>

#include "trapezoid_core.h"

/* 1/sqrt(2) - used by norm_cdf via erfc */
#ifndef TRAPZ_SQRT1_2
#define TRAPZ_SQRT1_2 0.70710678118654752440
#endif

/* ----
* normal cdf approximation: phi(x) = 0.5 * erfc(-x / sqrt(2))
* ---- */
double norm_cdf(double x) {
    return 0.5 * erfc(-x * TRAPZ_SQRT1_2);
}


/* ----
* Black-Scholes option price function
* cp = +1 for calls (OPT_CALL), -1 for puts (OPT_PUT)
* price = cp * (S * N(cp * d1) - K * e^{-rT} * N(cp * d2))
* ---- */
double bs_price(
    double S,
    double K,
    double r,
    double T,
    double sigma,
    int flag
)
{
    double cp = (flag == OPT_CALL) ? 1.0 : -1.0;
    double sqrtT = sqrt(T);
    double d1 = (log (S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
    double d2 = d1 - sigma * sqrtT;
    double disc = exp(-r * T);

    return cp * (S * norm_cdf(cp * d1) - K * disc * norm_cdf(cp * d2));
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


/* ----
* Sort a struct of (strike, implied vol) pairs so that calls and puts are independently
* sorted in ascending strike order. This is required for the trapezoidal integration to
* work correctly.
* The natural integration order for puts should go from the strike nearest the spot
* outward, mirroring how calls sweep upward from spot.
* ---- */
typedef struct {
    double strike;
    double ivol;
    double option_price;
} OptionRec;

static int cmp_strike_asc(const void *a, const void *b) {
    double ka = ((const OptionRec *)a)->strike;
    double kb = ((const OptionRec *)b)->strike;
    return (ka > kb) - (ka < kb);
}

static int cmp_strike_desc(const void *a, const void *b) {
    double ka = ((const OptionRec *)a)->strike;
    double kb = ((const OptionRec *)b)->strike;
    return (ka < kb) - (ka > kb);
}

/* ----
* Helper function to compute the trapezoidal sum for either calls or puts
* ---- */
static void trapz_leg(
    const OptionRec *recs,
    size_t n_leg,
    double spot,
    double (*kV)(double, double),
    double (*kW)(double, double),
    double (*kX)(double, double),
    double *out_V,
    double *out_W,
    double *out_X
)
{
    double V = 0.0, W = 0.0, X = 0.0;

    for (size_t j = 0; j < n_leg; ++j) {
        double K = recs[j].strike;
        double p = recs[j].option_price;
        double kv = kV(spot, K);
        double kw = kW(spot, K);
        double kx = kX(spot, K);
        double dK;

        if (j == 0) {
            /* Left endpoint */
            dK = fabs(K - spot);
            V += kv * p * dK;
            W += kw * p * dK;
            X += kx * p * dK;
        }
        else {
            /* Interior / right points */
            double Kp = recs[j-1].strike;
            double pp = recs[j-1].option_price;
            double kvp = kV(spot, Kp);
            double kwp = kW(spot, Kp);
            double kxp = kX(spot, Kp);
            dK = fabs(K - Kp);
            V += 0.5 * (kv * p + kvp * pp) * dK;
            W += 0.5 * (kw * p + kwp * pp) * dK;
            X += 0.5 * (kx * p + kxp * pp) * dK;
        }
    }

    *out_V = V;
    *out_W = W;
    *out_X = X;
}


/* ----
* Main public function to compute risk-neutral moments from option prices
* ---- */
int trapz_moments(
    size_t n,
    const double *strikes,
    const double *ivols,
    const int *flags,
    double spot,
    double r,
    double T,
    double *out_var,
    double *out_skew,
    double *out_kurt
)
{
    /* Default output to NaN */
    *out_var = *out_skew = *out_kurt = (double)NAN;

    /* This check is likely redundant since we check for at least 4 options in cython ext */
    if (n < 4) {
        return TRAPZ_ERR_FEW_OPT;
    }

    /* Count calls and puts */
    size_t n_calls = 0, n_puts= 0;
    for (size_t i = 0; i< n; ++i) {
        if (flags[i] == OPT_CALL){
            n_calls++;
        }
        else {
            n_puts++;
        }
    }

    if (n_calls == 0) {
        return TRAPZ_ERR_NO_CALLS;
    }
    if (n_puts == 0) {
        return TRAPZ_ERR_NO_PUTS;
    }

    /* Allocate arrays */
    OptionRec *calls = (OptionRec *)malloc(n_calls * sizeof(OptionRec));
    OptionRec *puts = (OptionRec *)malloc(n_puts * sizeof(OptionRec));

    if (!calls || !puts) {
        free(calls);
        free(puts);
        return TRAPZ_ERR_ALLOC;
    }

    /* Split into call-put buffers */
    size_t ic = 0, ip = 0;
    for (size_t i =0; i < n; ++i) {
        if (flags[i] == OPT_CALL) {
            calls[ic].strike = strikes[i];
            calls[ic].ivol = ivols[i];
            ++ic;
        }
        else {
            puts[ip].strike = strikes[i];
            puts[ip].ivol = ivols[i];
            ++ip;
        }
    }

    /* Sort ascending by strikes */
    qsort(calls, n_calls, sizeof(OptionRec), cmp_strike_asc);
    qsort(puts, n_puts, sizeof(OptionRec), cmp_strike_desc);

    /* Calc option price */
    for (size_t j = 0; j < n_calls; ++j) {
        calls[j].option_price = bs_price(
            spot,
            calls[j].strike,
            r,
            T,
            calls[j].ivol,
            OPT_CALL
        );
    }
    for (size_t j = 0; j < n_puts; ++j) {
        puts[j].option_price = bs_price(
            spot,
            puts[j].strike,
            r,
            T,
            puts[j].ivol,
            OPT_PUT
        );
    }

    /* Trapezoidal integration for calls and puts + contract payoffs */
    double Vc, Wc, Xc;
    double Vp, Wp, Xp;

    trapz_leg(calls, n_calls, spot, vc, wc, xc, &Vc, &Wc, &Xc);
    trapz_leg(puts,  n_puts,  spot, vp, wp, xp, &Vp, &Wp, &Xp);

    double V = Vc + Vp;
    double W = Wc + Wp;
    double X = Xc + Xp;

    /* compute moments */
    double exprt = exp(r * T);
    double muu = bkm_mu(r, T, V, W, X);
    double ev_mu2 = exprt * V - muu * muu;

    *out_var = ev_mu2;
    *out_skew = -(exprt * W - 3.0 * muu * exprt * V + 2.0 * muu * muu * muu) / pow(ev_mu2, 1.5);
    *out_kurt = (exprt * X
                - 4.0 * muu * exprt * W
                + 6.0 * exprt * muu * muu * V
                - 3.0 * muu * muu * muu * muu)
                / (ev_mu2 * ev_mu2);

    free(calls);
    free(puts);
    return TRAPZ_OK;
}
