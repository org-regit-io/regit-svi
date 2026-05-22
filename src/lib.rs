// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Arbitrage-free SVI volatility surfaces in pure Rust.
//!
//! `regit-svi` parametrises the implied volatility smile and surface with the
//! Stochastic Volatility Inspired (SVI) family — Raw SVI (Gatheral 2004),
//! SVI Jump-Wings, and the Surface SVI (SSVI) of Gatheral & Jacquier (2014) —
//! together with calibration to market quotes and explicit static-arbitrage
//! checks (butterfly and calendar-spread).
//!
//! Designed for auditability: every formula is hand-rolled from primary paper
//! sources with no external dependencies. A regulator, quant auditor, or new
//! engineer can trace every number to a citable derivation in [`MATH.md`].
//!
//! [`MATH.md`]: https://github.com/org-regit-io/regit-svi/blob/main/MATH.md
//!
//! # Quick start
//!
//! ```
//! use regit_svi::{Quote, calibration::quasi_explicit};
//!
//! // Market quotes for one maturity: (log-moneyness, total variance, weight).
//! let quotes = [
//!     Quote::new(-0.20, 0.0512, 1.0).unwrap(),
//!     Quote::new(-0.10, 0.0432, 1.0).unwrap(),
//!     Quote::new( 0.00, 0.0400, 1.0).unwrap(),
//!     Quote::new( 0.10, 0.0420, 1.0).unwrap(),
//!     Quote::new( 0.20, 0.0480, 1.0).unwrap(),
//! ];
//!
//! // Calibrate a raw SVI slice (quasi-explicit, de Marco-Martini).
//! let fit = quasi_explicit::calibrate(&quotes).unwrap();
//!
//! // Evaluate total variance and implied volatility anywhere on the slice.
//! let w = fit.slice.total_variance(0.05);
//! let vol = fit.slice.implied_vol(0.05, 1.0).unwrap();
//! assert!(w > 0.0 && vol > 0.0);
//!
//! // Certify the slice is free of butterfly arbitrage.
//! assert!(fit.butterfly_free);
//! ```
//!
//! # Architecture
//!
//! ```text
//! types          log-moneyness, total variance, market quotes (Quote)
//! errors         typed errors for parametrisation, conversion, calibration
//! math           numerical primitives — Nelder-Mead, Levenberg-Marquardt,
//!                linear least-squares (Cholesky), Brent root-finder
//!
//! raw            Raw SVI w(k) = a + b(rho(k-m) + sqrt((k-m)^2 + sigma^2))
//! jw             SVI Jump-Wings parametrisation (trader-facing parameters)
//! ssvi           Surface SVI w(k, theta) — arbitrage-free whole-surface form
//! convert        conversions between Raw, Jump-Wings, and SSVI slices
//!
//! arbitrage      butterfly (g(k) >= 0) and calendar-spread checks
//! density        risk-neutral density implied by a slice
//!
//! calibration/
//!   quasi_explicit   de Marco-Martini / Zeliade quasi-explicit slice fit
//!   least_squares    direct Levenberg-Marquardt slice fit
//!   ssvi             joint SSVI surface calibration
//!
//! surface        multi-slice surface assembly and interpolation
//! ```
//!
//! Part of [Regit OS](https://www.regit.io) — the operating system for
//! investment products. From Luxembourg.

#![forbid(unsafe_code)]

pub mod arbitrage;
pub mod calibration;
pub mod convert;
pub mod density;
pub mod errors;
pub mod jw;
pub mod math;
pub mod raw;
pub mod ssvi;
pub mod surface;
pub mod types;

// ─── Re-exports for ergonomic top-level access ─────────────────────────────

pub use arbitrage::{ButterflyReport, CalendarReport};
pub use calibration::CalibrationResult;
pub use convert::{jw_to_raw, raw_to_jw, ssvi_to_raw};
pub use density::DensityReport;
pub use errors::{CalibrationError, ConvertError, ParamError};
pub use jw::SviJw;
pub use raw::RawSvi;
pub use ssvi::{Phi, Ssvi};
pub use surface::Surface;
pub use types::{Quote, log_moneyness, quotes_from_triples, total_variance_from_vol};
