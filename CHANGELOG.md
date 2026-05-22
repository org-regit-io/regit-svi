<!-- Copyright 2026 Regit.io — Nicolas Koenig -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — 2026-05-22

First stable release. The public API is frozen under semantic versioning.

### Added — library implementation

The full SVI library lands in this release. Every formula is hand-rolled from
the primary paper sources cited in [`MATH.md`](MATH.md); the crate has zero
runtime dependencies and compiles cleanly to `wasm32-unknown-unknown`.

#### Core types (`src/types.rs`)
- `Quote` struct — a market quote `(k, w, weight)`: log-moneyness, observed
  total implied variance, non-negative fitting weight; `new` (validated) and
  `new_unchecked` constructors, `implied_vol(t)` accessor
- `quotes_from_triples`, `total_variance_from_vol`, `log_moneyness` — slice
  builders and conversion helpers

#### Typed errors (`src/errors.rs`)
- `ParamError` — invalid SVI / SSVI parameters or out-of-domain quotes
  (negative slope, correlation out of range, non-positive sigma / theta,
  negative minimum variance, invalid `phi` parameter, non-finite input)
- `ConvertError` — a parametrisation conversion with no valid pre-image
  (`|beta| > 1`, negative wing slope, degenerate Jump-Wings tuple)
- `CalibrationError` — calibration failures (too few quotes, empty quotes,
  non-convergence, all weights zero)
- All three implement `Display`, `Debug`, `std::error::Error`, with `From`
  conversions and `source()` chaining; `Copy` throughout

#### Numerical primitives (`src/math.rs`)
- `nelder_mead` — downhill-simplex minimisation (Nelder & Mead 1965) with the
  standard `(1, 2, 0.5, 0.5)` coefficients and a combined objective-spread /
  simplex-diameter convergence test
- `brent_root` — bracketed root-finding (Brent 1973) combining bisection,
  secant, and inverse quadratic interpolation
- `solve_spd` / `solve_spd_3` — symmetric positive-definite linear solves by
  Cholesky decomposition
- `levenberg_marquardt` — damped Gauss-Newton for nonlinear least-squares
  (Levenberg 1944; Marquardt 1963) with the gain-ratio damping update
- `index_to_f64` — lossless `usize -> f64` conversion via `u32` halves

#### Raw SVI (`src/raw.rs`)
- `RawSvi` struct — five-parameter slice `{a, b, rho, m, sigma}`; `new`
  (domain-validated) and `new_unchecked` constructors, `validate`
- `total_variance`, `w_prime`, `w_double_prime` — closed-form `w(k)` and its
  first two derivatives (MATH.md §3)
- `implied_vol`, `k_min`, `w_min`, `atm_total_variance`, `atm_skew`,
  `atm_curvature`, `left_wing_slope`, `right_wing_slope`

#### SVI Jump-Wings (`src/jw.rs`)
- `SviJw` struct — trader-facing parametrisation `{v_t, psi_t, p_t, c_t,
  v_tilde_t}` tagged with maturity `t`; `new` / `new_unchecked` / `validate`
- `atm_total_variance`, `min_total_variance`, `atm_vol` accessors

#### Surface SVI (`src/ssvi.rs`)
- `Phi` enum — smoothing function, `Heston { lambda }` and
  `PowerLaw { eta, gamma }`, with validated constructors and `eval`
- `Ssvi` struct — global correlation plus `phi`; `total_variance(k, theta)`,
  `slice_at(theta) -> RawSvi` (closed-form SSVI -> Raw map, MATH.md §6)
- `is_butterfly_free` / `is_butterfly_free_at` — Theorem 4.2 sufficient
  no-butterfly inequalities; `is_calendar_free` / `is_calendar_free_at` —
  Theorem 4.1 no-calendar conditions

#### Conversions (`src/convert.rs`)
- `raw_to_jw` — forward map Raw -> Jump-Wings (MATH.md §4)
- `jw_to_raw` — inverse map Jump-Wings -> Raw with the `|beta| <= 1` existence
  check, the genuine `m = 0` branch (`beta = 0`), and detection of the
  ambiguous vertex-at-ATM configuration
- `ssvi_to_raw` — SSVI slice at fixed `theta` to a raw SVI slice

#### Arbitrage checks (`src/arbitrage.rs`)
- `g` — the butterfly function `g(k)` in closed form (MATH.md §7)
- `wing_bound_ok` — the Lee wing bound `b*(1 + |rho|) <= 2`
- `butterfly_scan` / `is_butterfly_free` — dense `g`-grid scan with Brent
  refinement of the arbitrage boundary; returns a `ButterflyReport`
- `calendar_scan` / `is_calendar_free` — pointwise `w(k, t2) >= w(k, t1)`
  monotonicity check across two slices; returns a `CalendarReport`

#### Risk-neutral density (`src/density.rs`)
- `d_minus`, `d_plus` — the Black `d` quantities
- `risk_neutral_density` — `p(k)` implied by a slice (Breeden-Litzenberger;
  MATH.md §9)
- `integral` — composite Simpson integration of `p`; `density_report` — a
  `DensityReport` with the integrated mass and the minimum density

#### Calibration (`src/calibration/`)
- `quasi_explicit::calibrate` — de Marco-Martini / Zeliade quasi-explicit
  calibrator: the inner convex QP in `(a, d, c)` is solved exactly over the
  Zeliade domain by 3x3 normal equations plus a full face / edge / vertex
  enumeration; the outer 2-D problem in `(m, sigma)` is multi-started
  Nelder-Mead
- `least_squares::calibrate` / `least_squares::refine` — direct
  Levenberg-Marquardt over the five raw parameters with the analytic Jacobian
  (MATH.md §11) and `exp` / `tanh` domain reparametrisation
- `calibrate_slice` — default pipeline: quasi-explicit fit then an LM polish,
  keeping the lower-RMSE result; `CalibrationResult` carries the slice, RMSE,
  and a butterfly-arbitrage certificate
- `ssvi::calibrate` — joint SSVI surface fit over `(rho, phi-params)` with the
  Theorem 4.1 / 4.2 no-arbitrage inequalities enforced by penalty;
  `SsviMaturity`, `SsviCalibration`, `PhiFamily` types

#### Surface (`src/surface.rs`)
- `Surface` — multi-slice surface, built `from_slices` or `from_ssvi`;
  `total_variance` / `implied_vol` with linear-in-`w` maturity interpolation
  and flat-vol extrapolation; `is_calendar_free` cross-slice check

#### Tests, benchmarks, examples
- **145 inline unit tests** across all modules — golden values, derivative
  finite-difference checks, the Vogt butterfly-arbitrage slice, calibration
  recovery, optimiser correctness on Rosenbrock and known roots
- **32 integration tests** (`tests/integration.rs`) in seven suites — golden,
  round-trip, arbitrage oracle, density, calibration recovery, surface
  assembly, and `proptest` invariants (`w >= 0`, `w'' > 0`, Raw->JW->Raw
  identity, SSVI Theorem 4.2 slices pass the butterfly scan)
- **81 doc-tests** — every public item carries a runnable example
- **Criterion benchmarks** (`benches/svi.rs`) for raw / SSVI evaluation,
  arbitrage checks, conversions, and calibration
- **`examples/quickstart.rs`** — end-to-end slice and surface workflow

#### Crate metadata
- `clippy::pedantic` clean across the workspace and all targets
- `#![forbid(unsafe_code)]` at the crate root; no `unwrap` / `expect` /
  `panic!` in library code
- Dev-dependencies `approx`, `proptest`, `criterion` and the `[[bench]]`
  target added to `Cargo.toml`; `deny.toml` licence allow-list extended to
  cover the permissive dev-dependency tree

[1.0.0]: https://github.com/org-regit-io/regit-svi/releases/tag/v1.0.0
