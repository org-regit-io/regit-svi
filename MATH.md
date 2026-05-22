<!-- Copyright 2026 Regit.io — Nicolas Koenig -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# MATH.md — regit-svi

> Full formula derivations for every algorithm in this crate. Each section maps
> to the source module named in its heading and cites the primary paper
> reference. All formulas are shown in plain-text notation using code blocks.
>
> Every formula stated here is implemented by that module — the crate is the
> executable form of this document, and each `src/` file carries the same
> derivations and citations in its own doc comments.

---

## Table of contents

1. [Total implied variance and log-moneyness](#total-implied-variance-and-log-moneyness--srctypesrs)
2. [Raw SVI parametrisation](#raw-svi-parametrisation--srcrawrs)
3. [Raw SVI derivatives and ATM quantities](#raw-svi-derivatives-and-atm-quantities--srcrawrs)
4. [SVI Jump-Wings parametrisation](#svi-jump-wings-parametrisation--srcjwrs)
5. [Parametrisation conversions](#parametrisation-conversions--srcconvertrs)
6. [SSVI — Surface SVI](#ssvi--surface-svi--srcssvirs)
7. [Butterfly arbitrage and the g function](#butterfly-arbitrage-and-the-g-function--srcarbitragers)
8. [Calendar-spread arbitrage](#calendar-spread-arbitrage--srcarbitragers)
9. [Risk-neutral density](#risk-neutral-density--srcdensityrs)
10. [Slice calibration — quasi-explicit](#slice-calibration--quasi-explicit--srccalibrationquasi_explicitrs)
11. [Slice calibration — direct least-squares](#slice-calibration--direct-least-squares--srccalibrationleast_squaresrs)
12. [SSVI surface calibration](#ssvi-surface-calibration--srccalibration)
13. [Surface assembly and interpolation](#surface-assembly-and-interpolation--srcsurfacers)
14. [Numerical primitives](#numerical-primitives--srcmathrs)
15. [Algorithm references](#algorithm-references)

---

## Total implied variance and log-moneyness — `src/types.rs`

**Source:** Gatheral, *The Volatility Surface* (Wiley, 2006), Chapter 3.

SVI parametrises one maturity slice at a time. Fix a time to expiry `T > 0`
and a forward price `F` for that maturity. For a strike `K`, define the
**log-moneyness**:

```
k = ln(K / F)
```

`k = 0` is at-the-money-forward; `k > 0` is out-of-the-money for calls.

Let `sigma_BS(k)` be the Black implied volatility of the option struck at `k`.
SVI does not parametrise `sigma_BS` directly — it parametrises the
**total implied variance**:

```
w(k) = sigma_BS(k)^2 * T
```

`w` is the natural object: it is additive in maturity for a flat surface,
the no-arbitrage conditions take their simplest form in `w`, and `w >= 0` is
the only domain requirement. Implied volatility is recovered by:

```
sigma_BS(k) = sqrt(w(k) / T)
```

A **market quote** is a triple `(k_i, w_i, weight_i)`: a log-moneyness, an
observed total variance, and a non-negative fitting weight (e.g. vega, or
inverse bid-ask spread). A **slice** is a set of quotes sharing one maturity.

---

## Raw SVI parametrisation — `src/raw.rs`

**Source:** Gatheral, J., "A parsimonious arbitrage-free implied volatility
parameterization with application to the valuation of volatility derivatives",
Global Derivatives & Risk Management, Madrid (2004); Gatheral & Jacquier,
"Arbitrage-free SVI volatility surfaces", *Quantitative Finance* 14(1):59-71
(2014), Section 3.1.

The **raw SVI** parametrisation of a single slice is:

```
w(k) = a + b * ( rho * (k - m) + sqrt( (k - m)^2 + sigma^2 ) )
```

with five parameters `chi_R = {a, b, rho, m, sigma}`:

| Parameter | Domain        | Role |
|-----------|---------------|------|
| `a`       | `a in R`      | Vertical translation — overall variance level |
| `b`       | `b >= 0`      | Slope of the wings — angle between asymptotes |
| `rho`     | `|rho| < 1`   | Counter-clockwise rotation — skew |
| `m`       | `m in R`      | Horizontal translation — shifts the smile |
| `sigma`   | `sigma > 0`   | ATM curvature — smoothness of the vertex |

### Asymptotes

As `k -> +/- infinity`, `sqrt((k-m)^2 + sigma^2) -> |k - m|`, so `w` is
asymptotically linear with slopes:

```
left wing  (k -> -inf):  dw/dk -> b * (rho - 1)   <= 0
right wing (k -> +inf):  dw/dk -> b * (rho + 1)   >= 0
```

### Non-negativity of variance

`w` attains its minimum at:

```
k_min = m - rho * sigma / sqrt(1 - rho^2)
w_min = w(k_min) = a + b * sigma * sqrt(1 - rho^2)
```

*Derivation.* Setting `w'(k) = 0` (see next section) gives
`(k-m)/sqrt((k-m)^2+sigma^2) = -rho`. Solving for `u = k - m` yields
`u = -rho*sigma/sqrt(1-rho^2)`, hence `sqrt(u^2+sigma^2) = sigma/sqrt(1-rho^2)`.
Substituting back, `w_min = a + b*sigma*(1-rho^2)/sqrt(1-rho^2)
= a + b*sigma*sqrt(1-rho^2)`.

Therefore the slice produces non-negative variance everywhere iff:

```
a + b * sigma * sqrt(1 - rho^2) >= 0
```

This, together with `b >= 0`, `|rho| < 1`, `sigma > 0`, is the **validity
domain** of raw SVI. It is necessary but not sufficient for absence of static
arbitrage — see Sections 7 and 8.

---

## Raw SVI derivatives and ATM quantities — `src/raw.rs`

**Source:** direct differentiation of the raw SVI formula.

Write `u = k - m` and `r = sqrt(u^2 + sigma^2)`. Then:

```
w(k)   = a + b * (rho*u + r)
w'(k)  = b * ( rho + u / r )
w''(k) = b * sigma^2 / r^3
```

Because `b >= 0` and `sigma > 0`, the second derivative satisfies
`w''(k) > 0` for all `k`: **a raw SVI slice is strictly convex in `k`.**
Convexity of `w` is necessary but not sufficient for absence of butterfly
arbitrage (Section 7).

### At-the-money quantities

Evaluating at `k = 0` (so `u = -m`, `r = sqrt(m^2 + sigma^2)`):

```
w(0)   = a + b * ( -rho*m + sqrt(m^2 + sigma^2) )           ATM total variance
w'(0)  = b * ( rho - m / sqrt(m^2 + sigma^2) )              ATM variance skew
w''(0) = b * sigma^2 / (m^2 + sigma^2)^(3/2)                ATM variance curvature
```

ATM implied volatility is `sigma_ATM = sqrt(w(0) / T)`.

---

## SVI Jump-Wings parametrisation — `src/jw.rs`

**Source:** Gatheral & Jacquier (2014), Section 3.2.

Raw parameters have no direct financial meaning — `m` and `sigma` are not
quantities a trader observes. The **Jump-Wings (JW)** parametrisation
re-expresses a slice in five quantities that are read off the market, and is
maturity-dependent (it carries `t` explicitly). With `w_t := w(0)` the ATM
total variance at maturity `t`:

| JW parameter | Meaning |
|--------------|---------|
| `v_t`        | ATM variance: `v_t = w(0) / t` |
| `psi_t`      | ATM skew: `psi_t = w'(0) / (2 * sqrt(w(0)))` |
| `p_t`        | Left (put) wing slope: `p_t = b * (1 - rho) / sqrt(w(0))` |
| `c_t`        | Right (call) wing slope: `c_t = b * (1 + rho) / sqrt(w(0))` |
| `v_tilde_t`  | Minimum variance: `v_tilde_t = w_min / t = (a + b*sigma*sqrt(1-rho^2)) / t` |

### Forward map (raw -> JW)

Given `chi_R = {a, b, rho, m, sigma}` and maturity `t`, with
`w0 = a + b*(-rho*m + sqrt(m^2+sigma^2))`:

```
v_t       = w0 / t
psi_t     = ( b / (2*sqrt(w0)) ) * ( rho - m / sqrt(m^2+sigma^2) )
p_t       = ( b / sqrt(w0) ) * (1 - rho)
c_t       = ( b / sqrt(w0) ) * (1 + rho)
v_tilde_t = ( a + b*sigma*sqrt(1-rho^2) ) / t
```

### Inverse map (JW -> raw)

Given `chi_J = {v_t, psi_t, p_t, c_t, v_tilde_t}` and maturity `t`, set
`w = v_t * t`. Then (Gatheral & Jacquier 2014, Section 3.2):

```
b   = ( sqrt(w) / 2 ) * (c_t + p_t)
rho = 1 - p_t * sqrt(w) / b                  [ = (c_t - p_t) / (c_t + p_t) ]
beta  = rho - 2 * psi_t * sqrt(w) / b
alpha = sign(beta) * sqrt( 1/beta^2 - 1 )
m   = ( (v_t - v_tilde_t) * t )
      / ( b * ( -rho + sign(alpha)*sqrt(1+alpha^2) - alpha*sqrt(1-rho^2) ) )
sigma = alpha * m
a   = v_tilde_t * t - b * sigma * sqrt(1 - rho^2)
```

**Existence.** `alpha` is real only if `|beta| <= 1`; a JW tuple with
`|beta| > 1` does not correspond to any raw SVI slice and the conversion
returns a typed error.

**Degenerate case `m = 0`.** When `v_t = v_tilde_t` the numerator of `m`
vanishes; the ATM point *is* the vertex. Then `sigma` is recovered directly
from `w - a = b * sigma` and `v_tilde_t*t = a + b*sigma*sqrt(1-rho^2)`, giving
`sigma = (v_t - v_tilde_t)*t / ( b*(1 - sqrt(1-rho^2)) )`, handled as a
separate branch.

---

## Parametrisation conversions — `src/convert.rs`

**Source:** Gatheral & Jacquier (2014), Sections 3.2 and 4.

Three parametrisations are supported; the conversions form this graph:

```
   Raw  <------------------>  Jump-Wings        (bijective, per Section 4)
    ^
    |  slice-at-fixed-theta
    |
  SSVI                                          (one-directional: SSVI -> Raw)
```

- **Raw <-> JW** — bijective for any maturity `t > 0`, subject to the
  existence condition `|beta| <= 1` on the JW side.
- **SSVI -> Raw** — every SSVI slice at fixed `theta` is a raw SVI slice; the
  closed-form map is given in Section 6.
- **Raw -> SSVI** is *not* defined: SSVI is a constrained three-parameter
  surface family, so a generic raw slice has no SSVI pre-image.

---

## SSVI — Surface SVI — `src/ssvi.rs`

**Source:** Gatheral & Jacquier (2014), Section 4.

SSVI parametrises the **entire surface** at once, as a function of
log-moneyness `k` and the ATM total variance `theta = theta_t = w(0, t)`:

```
w(k, theta) = (theta / 2)
            * ( 1 + rho*phi(theta)*k
                  + sqrt( (phi(theta)*k + rho)^2 + (1 - rho^2) ) )
```

The free objects are:

- a single global correlation `rho in (-1, 1)`;
- the ATM total-variance term structure `theta_t`, a non-decreasing function
  of maturity (typically interpolated from observed ATM quotes);
- a smoothing function `phi : R+ -> R+`.

### Standard choices of phi

```
Heston-like:  phi(theta) = (1 / (lambda*theta))
                         * ( 1 - (1 - exp(-lambda*theta)) / (lambda*theta) )

Power-law:    phi(theta) = eta / ( theta^gamma * (1 + theta)^(1 - gamma) )
              with  eta > 0,  gamma in (0, 1)
```

### SSVI slice is a raw SVI slice

For fixed `theta`, writing `phi = phi(theta)`, the SSVI slice equals the raw
SVI slice with:

```
a     = (theta / 2) * (1 - rho^2)
b     = theta * phi / 2
rho   = rho                       (unchanged)
m     = -rho / phi
sigma = sqrt(1 - rho^2) / phi
```

*Derivation.* Factor `phi` out of the square root:
`sqrt((phi*k+rho)^2 + (1-rho^2)) = phi*sqrt((k + rho/phi)^2 + (1-rho^2)/phi^2)`.
Matching the linear term fixes `b = theta*phi/2` and `m = -rho/phi`; matching
the radius fixes `sigma = sqrt(1-rho^2)/phi`. The leftover constant
`b*rho*(-m) = theta*rho^2/2` is absorbed into `a`, so
`a = theta/2 - theta*rho^2/2 = (theta/2)(1-rho^2)`.

### No-arbitrage conditions for SSVI

SSVI's value is that static arbitrage reduces to closed-form conditions on
`rho` and `phi` (Gatheral & Jacquier 2014).

**No calendar-spread arbitrage (Theorem 4.1).** The surface is free of
calendar-spread arbitrage iff for every `theta > 0`:

```
(i)   d(theta_t) / dt >= 0
(ii)  0 <= d( theta * phi(theta) ) / d(theta)
           <= (1/rho^2) * (1 + sqrt(1 - rho^2)) * phi(theta)
```

(The upper bound in (ii) is `+infinity` when `rho = 0`.)

**No butterfly arbitrage (Theorem 4.2, sufficient).** The surface is free of
butterfly arbitrage if for every `theta > 0`:

```
(i)   theta * phi(theta)      * (1 + |rho|) < 4
(ii)  theta * phi(theta)^2    * (1 + |rho|) <= 4
```

**Specialised corollaries.**
- Heston-like `phi` is free of butterfly arbitrage when `lambda >= (1+|rho|)/4`.
- Power-law `phi` is free of butterfly arbitrage when `eta*(1+|rho|) <= 2`.

These inequalities are checked directly by `src/arbitrage.rs` for a given
`(rho, phi)` and enforced as constraints during SSVI calibration (Section 12).

---

## Butterfly arbitrage and the g function — `src/arbitrage.rs`

**Source:** Gatheral & Jacquier (2014), Section 2.2; Roper, M.,
"Arbitrage free implied volatility surfaces", preprint (2010).

A slice admits **butterfly arbitrage** when the risk-neutral density it
implies is negative somewhere — i.e. a butterfly spread has negative cost.
Define the function `g : R -> R`:

```
g(k) = ( 1 - k*w'(k) / (2*w(k)) )^2
     - ( w'(k) / 2 )^2 * ( 1/w(k) + 1/4 )
     + w''(k) / 2
```

A slice is **free of butterfly arbitrage** iff both hold:

```
(1)  g(k) >= 0          for all k in R
(2)  lim_{k -> +inf} d_plus(k) = -inf
```

where `d_plus(k) = -k/sqrt(w(k)) + sqrt(w(k))/2` (Section 9). Condition (2)
guarantees call prices vanish at infinite strike; for raw SVI it is
equivalent to the **wing bound**:

```
b * (1 + rho) <= 2     (right wing)
b * (1 - rho) <= 2     (left wing)   =>   b * (1 + |rho|) <= 2
```

This wing bound is also implied by Lee's moment formula (Lee 2004): implied
total variance cannot grow faster than `2|k|` asymptotically.

**Implementation.** `g(k)` is evaluated in closed form from `w, w', w''`
(Section 3). The check scans `g` on a dense grid of `k` spanning the quoted
range plus a wing margin, refining any sign change with a Brent root-find
(Section 14) to locate and report the arbitrage interval. For SSVI the
closed-form Theorem 4.2 conditions are used instead and are exact.

---

## Calendar-spread arbitrage — `src/arbitrage.rs`

**Source:** Gatheral & Jacquier (2014), Section 2.1.

Two slices at maturities `t_1 < t_2` admit **calendar-spread arbitrage** when
their total-variance curves cross: a longer-dated option would be cheaper than
a shorter-dated one with the same moneyness. Absence of calendar-spread
arbitrage is the pointwise monotonicity:

```
w(k, t_1) <= w(k, t_2)     for all k,   whenever t_1 < t_2
```

Both `w(k, t_1)` and `w(k, t_2)` are measured in each maturity's own
log-moneyness (against that maturity's forward).

**Implementation.** For a pair of raw slices the difference
`D(k) = w(k, t_2) - w(k, t_1)` is checked for non-negativity on a dense `k`
grid, with Brent refinement of any crossing. For SSVI the condition collapses
to Theorem 4.1 (Section 6) and is checked in closed form.

---

## Risk-neutral density — `src/density.rs`

**Source:** Breeden & Litzenberger (1978); Gatheral & Jacquier (2014), eq. (2.2).

The function `g(k)` is exactly the (normalised) risk-neutral probability
density in log-strike space. With:

```
d_minus(k) = -k / sqrt(w(k)) - sqrt(w(k)) / 2
d_plus(k)  = -k / sqrt(w(k)) + sqrt(w(k)) / 2
```

the risk-neutral density of the log-strike is:

```
p(k) = g(k) / sqrt( 2*pi*w(k) ) * exp( -d_minus(k)^2 / 2 )
```

`p(k) >= 0` for all `k` is equivalent to `g(k) >= 0`, which is why the
butterfly check (Section 7) is a density-positivity check. `p` integrates to
1 over `k in R` for an arbitrage-free slice; the implementation exposes `p`
and a numerical integral as a diagnostic.

---

## Slice calibration — quasi-explicit — `src/calibration/quasi_explicit.rs`

**Source:** De Marco, S. & Martini, C., "Quasi-explicit calibration of
Gatheral's SVI model", Zeliade Systems White Paper ZWP-0005 (2009).

Direct least-squares over all five raw parameters is non-convex and sensitive
to the starting point. The quasi-explicit method removes that fragility by
exploiting a change of variables that makes the problem **linear in three of
the five parameters**.

### Reduction

Fix the two "nonlinear" parameters `m` and `sigma`. Substitute
`y = (k - m) / sigma`, so `sqrt((k-m)^2 + sigma^2) = sigma*sqrt(y^2 + 1)`:

```
w = a + b*rho*sigma*y + b*sigma*sqrt(y^2 + 1)
```

Define `c = b*sigma` and `d = rho*b*sigma`. The model becomes **affine** in
`(a, d, c)`:

```
w(y) = a + d*y + c*sqrt(y^2 + 1)
```

### Inner problem — convex, solved exactly

For fixed `(m, sigma)`, minimise the weighted residual over `(a, d, c)`:

```
f(a, d, c) = sum_i  weight_i * ( a + d*y_i + c*sqrt(y_i^2 + 1) - w_i )^2
```

on the convex domain `D` (Zeliade 2009):

```
0 <= c <= 4*sigma
|d| <= c            and      |d| <= 4*sigma - c
0 <= a <= max_i w_i
```

`|d| <= c` enforces `|rho| <= 1`; together with `|d| <= 4*sigma - c` it gives
`c + |d| <= 4*sigma`, i.e. `b*(1+|rho|) <= 4` — the deliberately generous
Zeliade search box, which contains the tighter Lee wing bound
`b*(1+|rho|) <= 2` (Section 7). `a >= 0` enforces non-negative variance.
`f` is a convex quadratic, so its minimum over the polytope `D` is
either the unconstrained stationary point (from the 3x3 normal equations) or,
if that is infeasible, lies on a face — found by solving the reduced
least-squares problem on each face, edge, and vertex of `D` and taking the
feasible minimiser. With three variables this enumeration is small and exact.

### Outer problem — two-dimensional

Let `f*(m, sigma)` be the optimal inner residual. Minimise:

```
g(m, sigma) = f*(m, sigma)        over  m in R,  sigma > 0
```

This 2-D, mildly non-convex problem is solved with the Nelder-Mead simplex
(Section 14), optionally multi-started across a small grid of `(m, sigma)`
seeds for robustness.

### Recovery

From the optimal `(a, c, d, m, sigma)`:

```
b   = c / sigma
rho = d / c          (b = 0 and rho = 0 when c = 0)
```

The result is a raw SVI slice plus the fit residual (RMSE) and a flag from the
butterfly check (Section 7).

---

## Slice calibration — direct least-squares — `src/calibration/least_squares.rs`

**Source:** Levenberg (1944); Marquardt (1963).

A direct calibrator is provided as an alternative and as a refinement step for
the quasi-explicit result. It minimises:

```
F(chi_R) = sum_i  weight_i * ( w_SVI(k_i; chi_R) - w_i )^2
```

over `chi_R = {a, b, rho, m, sigma}` with the Levenberg-Marquardt algorithm.
The five partial derivatives needed for the Jacobian are closed-form
(`u = k - m`, `r = sqrt(u^2 + sigma^2)`):

```
dw/da     = 1
dw/db     = rho*u + r
dw/drho   = b*u
dw/dm     = b * ( -rho - u/r )
dw/dsigma = b * sigma / r
```

No finite differences are used. Domain constraints (`b >= 0`, `|rho| < 1`,
`sigma > 0`, `a + b*sigma*sqrt(1-rho^2) >= 0`) are imposed by smooth
reparametrisation: `b = exp(b_hat)`, `sigma = exp(sigma_hat)`,
`rho = tanh(rho_hat)`. LM damping `mu` is adapted by the standard gain-ratio
rule; convergence is declared on a small gradient norm or step size.

A good initial guess matters for direct LM; the default is the quasi-explicit
solution (Section 10), making the two calibrators complementary —
quasi-explicit for a robust global fit, LM for a fast local polish.

---

## SSVI surface calibration — `src/calibration/`

**Source:** Gatheral & Jacquier (2014), Sections 4-5.

SSVI calibration fits the whole surface jointly so that the result is
arbitrage-free by construction:

1. **ATM term structure.** From the ATM quote of each maturity, build the
   non-decreasing `theta_t` curve (monotone interpolation; Section 13).
2. **Global fit.** Minimise the total weighted residual across all slices over
   `rho` and the `phi` parameters (`eta, gamma` for power-law, or `lambda` for
   Heston-like), with `w(k, theta)` from Section 6.
3. **Constraints.** The Theorem 4.1 and 4.2 inequalities (Section 6) are
   enforced throughout, so the fitted surface is free of both butterfly and
   calendar-spread arbitrage. The outer search is 2- or 3-D (Nelder-Mead).

This yields fewer parameters than independent per-slice fits and a surface
that needs no post-hoc arbitrage repair.

---

## Surface assembly and interpolation — `src/surface.rs`

**Source:** Gatheral, *The Volatility Surface* (2006), Chapter 3.

A **surface** is an ordered set of calibrated slices at maturities
`t_1 < ... < t_n`. To evaluate total variance at an arbitrary `(k, T)`:

- **Maturity inside the grid.** Locate the bracketing slices `t_j <= T < t_{j+1}`
  and interpolate **linearly in total variance along constant `k`**:

  ```
  w(k, T) = ( (t_{j+1} - T) * w(k, t_j) + (T - t_j) * w(k, t_{j+1}) )
            / (t_{j+1} - t_j)
  ```

  Linear interpolation in `w` is monotone in `T`, so it introduces no
  calendar-spread arbitrage provided the bracketing slices are themselves
  ordered (`w(k, t_j) <= w(k, t_{j+1})`).

- **Maturity outside the grid.** Total variance is extrapolated flat in
  `sigma_BS` (constant implied volatility beyond the first/last slice), the
  conservative market default.

For an SSVI-backed surface, evaluation is direct from the closed form in
Section 6 at the interpolated `theta_T`, and no per-`k` interpolation is
needed. Implied volatility is `sigma_BS(k, T) = sqrt(w(k, T) / T)`.

---

## Numerical primitives — `src/math.rs`

**Source:** Nelder & Mead (1965); Levenberg (1944), Marquardt (1963);
Brent (1973).

The crate is zero-dependency, so every optimiser and solver is hand-rolled
from its primary source. All are pure functions, deterministic, and `std`-only.

### Nelder-Mead downhill simplex

Gradient-free minimisation for the low-dimensional outer problems (2-D
quasi-explicit, 2-3-D SSVI). Maintains a simplex of `n+1` vertices and applies
reflection, expansion, contraction, and shrink steps with the standard
coefficients `(1, 2, 0.5, 0.5)`. Terminates on simplex diameter or function
spread below tolerance. Multi-start support guards against local minima.

### Levenberg-Marquardt

Damped Gauss-Newton for nonlinear least-squares (direct slice calibration).
Solves `(J^T W J + mu*diag(J^T W J)) delta = -J^T W r` each iteration, with
`mu` adapted by the gain ratio between actual and predicted residual
reduction. The Jacobian `J` is analytic (Section 11).

### Linear least-squares

Exact solver for the quasi-explicit inner problem: forms the `3x3` weighted
normal-equations system and solves it by Cholesky decomposition, with the
constrained-domain face enumeration of Section 10.

### Brent root-finder

Bracketed root-finding combining bisection, secant, and inverse quadratic
interpolation (Brent 1973). Used to refine sign changes of `g(k)` (butterfly)
and of the slice difference `D(k)` (calendar) to a reported tolerance.
Guaranteed to converge on any sign-changing bracket.

---

## Algorithm references

| Algorithm | Primary reference |
|---|---|
| Raw SVI parametrisation | Gatheral, J., "A parsimonious arbitrage-free implied volatility parameterization", Global Derivatives, Madrid (2004) |
| Volatility surface (total variance, conventions) | Gatheral, J., *The Volatility Surface: A Practitioner's Guide*, Wiley (2006) |
| SVI Jump-Wings, SSVI, arbitrage conditions | Gatheral, J. & Jacquier, A., "Arbitrage-free SVI volatility surfaces", *Quantitative Finance* 14(1):59-71 (2014) |
| Quasi-explicit calibration | De Marco, S. & Martini, C., "Quasi-explicit calibration of Gatheral's SVI model", Zeliade Systems White Paper ZWP-0005 (2009) |
| Moment formula / wing slope bound | Lee, R. W., "The moment formula for implied volatility at extreme strikes", *Mathematical Finance* 14(3):469-480 (2004) |
| Arbitrage-free surface conditions | Roper, M., "Arbitrage free implied volatility surfaces", preprint, University of Sydney (2010) |
| Risk-neutral density from option prices | Breeden, D. & Litzenberger, R., "Prices of state-contingent claims implicit in option prices", *Journal of Business* 51(4):621-651 (1978) |
| Nelder-Mead simplex | Nelder, J. A. & Mead, R., "A simplex method for function minimization", *The Computer Journal* 7(4):308-313 (1965) |
| Levenberg-Marquardt | Levenberg, K., *Quarterly of Applied Mathematics* 2(2):164-168 (1944); Marquardt, D. W., *J. SIAM* 11(2):431-441 (1963) |
| Brent's method | Brent, R. P., *Algorithms for Minimization Without Derivatives*, Prentice-Hall (1973) |

---

*Part of [Regit OS](https://www.regit.io) — the operating system for investment products. From Luxembourg.*
