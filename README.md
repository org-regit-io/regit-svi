<!-- Copyright 2026 Regit.io — Nicolas Koenig -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# regit-svi

Arbitrage-free SVI volatility surfaces. Zero-dependency, pure Rust.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.85%2B-orange.svg)](https://www.rust-lang.org)

## What it does

`regit-svi` parametrises the implied volatility smile and surface with the
**Stochastic Volatility Inspired (SVI)** family, calibrates it to market
quotes, and certifies the result free of static arbitrage.

It covers three parametrisations — **Raw SVI** (Gatheral 2004), **SVI
Jump-Wings**, and the **Surface SVI / SSVI** of Gatheral & Jacquier (2014) —
two calibration engines, and explicit **butterfly** and **calendar-spread**
arbitrage checks.

Every formula is hand-rolled from primary paper sources with no external
dependencies. A regulator, quant auditor, or new engineer can open any source
file and trace every number to a citable derivation in [MATH.md](MATH.md).

## Why this crate exists

A volatility surface is the input to every option pricer and every risk
engine. Markets quote implied volatilities at a sparse, discrete grid of
strikes and maturities — but pricing and risk need a *continuous* surface.

The naive fix is to interpolate. **Interpolation silently injects static
arbitrage.** Spline a smile through five quotes and the implied risk-neutral
density goes negative between them — a butterfly spread with negative cost.
Interpolate across maturities independently and the total-variance curves
cross — a calendar spread that pays to hold. A surface with either defect
produces mispriced exotics, unstable Greeks, and risk numbers that cannot be
trusted, and the defect is invisible unless you test for it.

SVI solves this at the parametrisation level. A **raw SVI** slice is a
five-parameter curve in total variance whose convexity and wing behaviour are
controlled by construction. **SSVI** goes further: it parametrises the whole
surface so that absence of butterfly *and* calendar-spread arbitrage reduces
to closed-form inequalities on three parameters — the surface is
arbitrage-free *by construction*, not by repair.

`regit-svi` implements that machinery, and — crucially — ships the
**verification** alongside it: the butterfly function `g(k)` and the
calendar-monotonicity test are exposed as first-class checks, so any surface,
however it was built, can be certified or flagged.

This sits within [Regit OS](https://www.regit.io): `regit-svi` is the surface
layer. It consumes implied volatilities recovered by
[`regit-blackscholes`](https://github.com/org-regit-io/regit-blackscholes) and
produces a clean, arbitrage-checked surface for pricing and risk downstream —
quant-proven math (Gatheral, Gatheral-Jacquier, primary-source derivations)
made auditable.

## Quick start

```toml
[dependencies]
regit-svi = "0.1"
```

```rust
use regit_svi::{Quote, calibration::quasi_explicit};

// Market quotes for one maturity: (log-moneyness, total variance, weight).
let quotes = [
    Quote::new(-0.20, 0.0512, 1.0).unwrap(),
    Quote::new(-0.10, 0.0432, 1.0).unwrap(),
    Quote::new( 0.00, 0.0400, 1.0).unwrap(),
    Quote::new( 0.10, 0.0420, 1.0).unwrap(),
    Quote::new( 0.20, 0.0480, 1.0).unwrap(),
];

// Calibrate a raw SVI slice (quasi-explicit, de Marco-Martini).
let fit = quasi_explicit::calibrate(&quotes).unwrap();
println!("RMSE: {:.2e}", fit.rmse);

// Evaluate total variance and implied volatility anywhere on the slice.
let w   = fit.slice.total_variance(0.05);
let vol = fit.slice.implied_vol(0.05, 1.0).unwrap(); // time to expiry = 1y

// The calibration result carries a butterfly-arbitrage certificate.
assert!(fit.butterfly_free);
```

See [`examples/quickstart.rs`](examples/quickstart.rs) for a complete working
example covering conversions, the SSVI surface fit, and the arbitrage checks.

## Parametrisations

| Parametrisation | Form | Use case | Reference |
|---|---|---|---|
| Raw SVI | `w(k) = a + b(ρ(k−m) + √((k−m)² + σ²))` | Math core; single-slice fitting | Gatheral (2004) |
| SVI Jump-Wings | `(vₜ, ψₜ, pₜ, cₜ, ṽₜ)` | Trader-facing — ATM vol/skew, wing slopes | Gatheral & Jacquier (2014) |
| SSVI | `w(k,θ) = θ/2·(1 + ρφk + √((φk+ρ)² + 1−ρ²))` | Whole surface, arbitrage-free by construction | Gatheral & Jacquier (2014) |

All conversions that exist between them are closed-form: Raw ↔ Jump-Wings is
bijective; every SSVI slice maps to a raw slice. See [MATH.md](MATH.md) §4–6.

## Calibration

Two engines, by design complementary:

| Engine | Method | Strength |
|---|---|---|
| Quasi-explicit | de Marco–Martini / Zeliade — 2-D outer search, closed-form convex inner solve | Robust; no sensitivity to the starting point |
| Direct least-squares | Levenberg–Marquardt over the 5 raw parameters, analytic Jacobian | Fast local polish from a good seed |

The default pipeline calibrates with the quasi-explicit method, then optionally
refines with Levenberg–Marquardt. SSVI surfaces are calibrated jointly across
maturities with the Theorem 4.1/4.2 no-arbitrage inequalities enforced
throughout. See [MATH.md](MATH.md) §10–12.

## Arbitrage checks

| Check | Condition | Module |
|---|---|---|
| Butterfly (no negative density) | `g(k) ≥ 0` for all `k` | `src/arbitrage.rs` |
| Calendar spread (variance monotone in `T`) | `w(k, t₁) ≤ w(k, t₂)` for `t₁ < t₂` | `src/arbitrage.rs` |
| SSVI butterfly / calendar | Closed-form inequalities on `(ρ, φ)` | `src/arbitrage.rs` |

The risk-neutral density `p(k)` implied by a slice is exposed directly for
inspection. See [MATH.md](MATH.md) §7–9.

## Architecture

```
src/
  lib.rs                       # Module declarations + re-exports
  types.rs                     # Log-moneyness, total variance, market quotes
  errors.rs                    # Typed errors — parametrisation, conversion, calibration
  math.rs                      # Nelder-Mead, Levenberg-Marquardt, linear LS, Brent

  raw.rs                       # Raw SVI: w(k), derivatives, ATM quantities
  jw.rs                        # SVI Jump-Wings parametrisation
  ssvi.rs                      # Surface SVI: w(k, θ), φ functions, no-arb conditions
  convert.rs                   # Raw <-> JW, SSVI -> Raw conversions

  arbitrage.rs                 # Butterfly g(k) + calendar-spread checks
  density.rs                   # Risk-neutral density from a slice

  calibration/
    mod.rs                     # Default pipeline + CalibrationResult
    quasi_explicit.rs          # de Marco-Martini / Zeliade quasi-explicit
    least_squares.rs           # Direct Levenberg-Marquardt
    ssvi.rs                    # Joint SSVI surface calibration

  surface.rs                   # Multi-slice surface assembly + interpolation
```

One file, one domain. Each function is pure, deterministic, and composable.

## Testing

```bash
cargo test                      # 258 tests
cargo run --example quickstart  # End-to-end slice + surface workflow
cargo bench                     # Criterion benchmarks
```

**145 unit tests** — golden values, finite-difference derivative checks,
parametrisation round-trips, the optimiser primitives (Nelder-Mead on
Rosenbrock, Brent on known roots), and calibration parameter recovery.

**32 integration tests** across seven suites: hand-computed golden anchors;
Raw ↔ Jump-Wings and SSVI → Raw round-trip identities; an arbitrage oracle
including the Axel Vogt slice (a documented butterfly-arbitrage example);
risk-neutral density mass and positivity; synthetic-data calibration recovery
for both engines and the SSVI surface fit; multi-slice surface assembly; and
`proptest` invariants (`w ≥ 0`, `w'' > 0`, Raw→JW→Raw identity, SSVI
Theorem 4.2 slices passing the butterfly scan).

**81 doc-tests** — every public item carries a runnable example.

## Code quality

- `#![forbid(unsafe_code)]` crate-wide
- `clippy::pedantic` with zero warnings
- Every public function documented with its mathematical reference
- No `unwrap()` or `panic!()` in library code — all failure paths typed
- Deterministic: same input produces bit-identical output
- WASM-clean: `cargo build --target wasm32-unknown-unknown` with no changes
- 258 tests — unit, integration, `proptest` invariants, doc-tests — plus
  `criterion` benchmarks (see [Testing](#testing))

## Dependencies

**Runtime: zero.** Only `std`. No `nalgebra`, no `argmin`, no `libm`, no FFI.
Every optimiser and solver — Nelder-Mead, Levenberg-Marquardt, linear
least-squares, Brent — is hand-rolled from its primary source.

License and supply-chain policy is enforced via `cargo-deny` (`deny.toml`).
No copyleft dependencies.

## Algorithms

All implemented from primary paper sources. No ports from Python, no reading
existing Rust crates.

| Algorithm | Reference |
|---|---|
| Raw SVI parametrisation | Gatheral, *A parsimonious arbitrage-free IV parameterization*, Madrid (2004) |
| Volatility surface conventions | Gatheral, *The Volatility Surface*, Wiley (2006) |
| SVI Jump-Wings, SSVI, arbitrage conditions | Gatheral & Jacquier, *Quantitative Finance* 14(1):59–71 (2014) |
| Quasi-explicit calibration | De Marco & Martini, Zeliade Systems White Paper ZWP-0005 (2009) |
| Moment formula / wing slope bound | Lee, *Mathematical Finance* 14(3):469–480 (2004) |
| Arbitrage-free surface conditions | Roper, *Arbitrage free implied volatility surfaces*, preprint (2010) |
| Risk-neutral density | Breeden & Litzenberger, *Journal of Business* 51(4):621–651 (1978) |
| Nelder-Mead simplex | Nelder & Mead, *The Computer Journal* 7(4):308–313 (1965) |
| Levenberg-Marquardt | Levenberg (1944); Marquardt, *J. SIAM* 11(2):431–441 (1963) |
| Brent's method | Brent, *Algorithms for Minimization Without Derivatives*, Prentice-Hall (1973) |

## Documentation

- [MATH.md](MATH.md) — Full mathematical derivations for every algorithm
- [CHANGELOG.md](CHANGELOG.md) — Release history
- [SECURITY.md](SECURITY.md) — Vulnerability disclosure policy

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

```
Copyright 2026 Regit.io — Nicolas Koenig
```

---

Part of [Regit OS](https://www.regit.io) — the operating system for investment products. From Luxembourg.
