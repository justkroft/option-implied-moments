<!-- Language switcher badges -->

[![English](https://img.shields.io/badge/English-blue)](./README.md)
[![日本語](https://img.shields.io/badge/日本語-red)](./README.ja.md)

# Option Implied Moments

This repository contains code to compute option implied, or risk-neutral, higher moments from stock option data. Stock return characteristics like volatility, skewness, and kurtosis are computed based on out-of-the-money stock option contracts under risk-neutral measure $\mathbb{Q}$.

Risk-neutral volatility, skewness, and kurtosis are estimated from a cross-section of out-of-the-money calls and puts following [[1]](#1):

$$
Var^{\mathbb{Q}} = \frac{e^{r\tau}V_{i,t} - \mu^{2}}{\tau},
$$
$$
Skew^{\mathbb{Q}} = \frac{e^{r\tau}W - 3\mu e^{r\tau}V + 2\mu^{3}}{[e^{r\tau}V-\mu^{2}]^{3/2}},
$$
$$
Kurt^{\mathbb{Q}} = \frac{e^{r\tau}X - 4\mu e^{r\tau}W + 6e^{r\tau}\mu^{2}V - 3\mu^{4}}{[e^{r\tau}V-\mu^{2}]^{2}} - 3
$$

where

$$
\mu = e^{r\tau} - 1 - \frac{e^{r\tau}}{2}V - \frac{e^{r\tau}}{6}W - \frac{e^{r\tau}}{24}X
$$

$V$, $W$, and $X$ are the expected squared, cubed and fourth-power log of the stock return during the period from $t$ to time $t + \tau$ under measure $\mathbb{Q}$; $r$ is the continuously compounded risk-free rate for the same period. To estimate these parameters, we follow the trapezoidal implementation from [[2]](#2) to use discrete strike prices from calls and put options. Please refer to Appendix B of the paper for the formalised implementation details.


## Programming Implementation Details

The library is implemented in three layers.

> Note: This library works best with implied volatility surface data from data providers like WRDS.

### C layer

All numerical work is done in pure C:

- **Normal CDF** — approximated using complementary error function:

$$
\Phi(x) = 0.5 \times \text{erfc}(\frac{-x}{\sqrt(2)})
$$

- **Black-Scholes pricing** — OTM option prices are computed from implied
  volatilities using the standard closed-form formula.
- **Trapezoidal integration** — calls and puts are sorted by strike, priced, and
  integrated separately using the trapezoidal rule following [[2]](#2).

### Cython layer

The Cython layer functions as a bridge between NumPy and the C extension. It accepts flat contiguous
NumPy arrays in a CSR (compressed sparse row) layout.
Strikes, implied volatilities, and flags concatenated across all groups,
with an `indptr` array marking the boundary of each group.

Groups are then processed in parallel using OpenMP.

### Python layer

The Python layer is responsible for data preparation only. It filters OTM
options, encodes call/put flags as integers, sorts and groups the data by stock
and time period, and assembles the CSR layout that the Cython layer
expects. After the Cython call returns, it reconstructs the output as a Polars
DataFrame.


# References
<a id="1">[1]</a>
Bakshi, G., Kapadia, N., & Madan, D. (2003). Stock return characteristics, skew laws, and the differential pricing of individual equity options. *The Review of Financial Studies, 16*(1), 101-143.

<a id="2">[2]</a>
Bali, Turan G. and Hu, Jianfeng and Murray, Scott, Option Implied Volatility, Skewness, and Kurtosis and the Cross-Section of Expected Stock Returns (January 1, 2019). Georgetown McDonough School of Business Research Paper, Available at SSRN: https://ssrn.com/abstract=2322945 or http://dx.doi.org/10.2139/ssrn.2322945

# Getting Started
*Prerequisites — you need CMake and a C compiler installed on your machine. The build process uses CMake to compile the C extension and link against system OpenMP. On macOS with Apple Clang, ensure you have Homebrew's LLVM installed.*

Clone the repository

```bash
git clone https://github.com/justkroft/option-implied-moments.git
cd option-implied-moments
```

and setup your virtual environment using `uv`.

```bash
uv venv  # create environment
uv sync  # sync all dependencies from toml file
```

Then, activate the environment:

```bash
source .venv/bin/activate  # on Mac
./venv/Scripts/activate    # on Windows
```

If `make` is available on your platform, you can build the extension as follows:

```bash
make install
make build
```

Else, you can make a local, editable install as follows:

```bash
uv pip install scikit-build-core numpy
uv pip install --no-build-isolation -e . -Ccmake.build-type=Release
```

## OpenMP Threading

The library uses OpenMP for parallel computation across groups. You can control the number of threads:

```python
from src.ext.omp_utils import get_max_threads, set_num_threads

max_threads = get_max_threads()
print(f"Available threads: {max_threads}")

set_num_threads(8)  # Use 8 threads
```

Please take a look at [the example notebook](example.ipynb) for an example of how to use the library.

# License
[MIT](LICENSE) License © 2026-PRESENT [justkroft](https://github.com/justkroft)
