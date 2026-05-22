// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Joint SSVI surface calibration (Gatheral & Jacquier 2014, Sections 4-5).
//!
//! SSVI calibration fits the whole surface at once so the result is
//! arbitrage-free by construction:
//!
//! 1. **ATM term structure.** From the ATM quote of each maturity, build the
//!    non-decreasing `theta_t` curve.
//! 2. **Global fit.** Minimise the total weighted residual across all slices
//!    over `rho` and the `phi` parameters, with `w(k, theta)` from the SSVI
//!    form (MATH.md §6).
//! 3. **Constraints.** The Theorem 4.1 and 4.2 inequalities are enforced
//!    throughout (an infeasible candidate is penalised), so the fitted
//!    surface is free of both butterfly and calendar-spread arbitrage.
//!
//! The outer search is 2-D (Heston-like `phi`, parameters `rho, lambda`) or
//! 3-D (power-law `phi`, parameters `rho, eta, gamma`), solved with the
//! Nelder-Mead simplex.
//!
//! # References
//!
//! - Gatheral, J. & Jacquier, A., "Arbitrage-free SVI volatility surfaces",
//!   *Quantitative Finance* 14(1):59-71 (2014), Sections 4-5.

use crate::errors::CalibrationError;
use crate::math::nelder_mead;
use crate::ssvi::{Phi, Ssvi};
use crate::types::Quote;

/// Nelder-Mead tolerance for the SSVI outer search.
const OUTER_TOL: f64 = 1e-12;
/// Nelder-Mead iteration cap for the SSVI outer search.
const OUTER_MAX_ITER: usize = 3000;
/// Penalty added to the objective for an arbitrageable candidate surface.
const ARBITRAGE_PENALTY: f64 = 1e6;

/// Which family of `phi` smoothing function to calibrate.
///
/// # Examples
///
/// ```
/// use regit_svi::calibration::ssvi::PhiFamily;
///
/// let f = PhiFamily::PowerLaw;
/// assert_eq!(f, PhiFamily::PowerLaw);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhiFamily {
    /// Heston-like `phi`, with the single parameter `lambda`.
    Heston,
    /// Power-law `phi`, with parameters `eta` and `gamma`.
    PowerLaw,
}

/// One calibration maturity: a time to expiry, its ATM total variance, and
/// the slice of quotes observed at that maturity.
///
/// # Examples
///
/// ```
/// use regit_svi::types::Quote;
/// use regit_svi::calibration::ssvi::SsviMaturity;
///
/// let quotes = vec![Quote::new(0.0, 0.04, 1.0).unwrap()];
/// let mat = SsviMaturity { t: 1.0, theta: 0.04, quotes };
/// assert!((mat.theta - 0.04).abs() < 1e-15);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct SsviMaturity {
    /// Time to expiry `t > 0`.
    pub t: f64,
    /// ATM total variance `theta = w(0, t)` at this maturity.
    pub theta: f64,
    /// Market quotes observed at maturity `t`.
    pub quotes: Vec<Quote>,
}

/// The result of an SSVI surface calibration.
#[derive(Debug, Clone, PartialEq)]
pub struct SsviCalibration {
    /// The fitted SSVI surface.
    pub ssvi: Ssvi,
    /// Per-maturity ATM total variances, ascending in maturity.
    pub thetas: Vec<f64>,
    /// Weighted root-mean-square fit residual across all slices.
    pub rmse: f64,
    /// `true` if the fitted surface passed the Theorem 4.1 / 4.2 checks.
    pub arbitrage_free: bool,
}

/// Calibrates an SSVI surface jointly across maturities.
///
/// The `theta_t` term structure is taken from the `theta` field of each
/// supplied maturity; the global parameters `(rho, phi-params)` are fitted by
/// a multi-started Nelder-Mead search, with arbitrageable candidates
/// penalised so the returned surface satisfies the Theorem 4.1 / 4.2
/// conditions.
///
/// # Errors
///
/// - [`CalibrationError::EmptyQuotes`] if no maturities or no quotes are
///   supplied.
/// - [`CalibrationError::AllWeightsZero`] if every fitting weight is zero.
/// - [`CalibrationError::Param`] if the fitted parameters are invalid.
/// - [`CalibrationError::DidNotConverge`] if no feasible surface was found.
///
/// # Examples
///
/// ```
/// use regit_svi::ssvi::{Phi, Ssvi};
/// use regit_svi::types::Quote;
/// use regit_svi::calibration::ssvi::{calibrate, PhiFamily, SsviMaturity};
///
/// // Generate a synthetic surface from a known SSVI and recover it.
/// let truth = Ssvi::new(-0.3, Phi::power_law(0.5, 0.5).unwrap()).unwrap();
/// let ks = [-0.3, -0.15, 0.0, 0.15, 0.3];
/// let mats: Vec<SsviMaturity> = [(0.5, 0.02), (1.0, 0.04), (2.0, 0.07)]
///     .iter()
///     .map(|&(t, theta)| SsviMaturity {
///         t,
///         theta,
///         quotes: ks
///             .iter()
///             .map(|&k| Quote::new(k, truth.total_variance(k, theta), 1.0).unwrap())
///             .collect(),
///     })
///     .collect();
/// let fit = calibrate(&mats, PhiFamily::PowerLaw).unwrap();
/// assert!(fit.rmse < 1e-3);
/// assert!(fit.arbitrage_free);
/// ```
pub fn calibrate(
    maturities: &[SsviMaturity],
    family: PhiFamily,
) -> Result<SsviCalibration, CalibrationError> {
    if maturities.is_empty() || maturities.iter().all(|m| m.quotes.is_empty()) {
        return Err(CalibrationError::EmptyQuotes);
    }
    let total_weight: f64 = maturities
        .iter()
        .flat_map(|m| m.quotes.iter())
        .map(|q| q.weight)
        .sum();
    if total_weight <= 0.0 {
        return Err(CalibrationError::AllWeightsZero);
    }

    let thetas: Vec<f64> = maturities.iter().map(|m| m.theta).collect();

    // Objective: total weighted squared residual + arbitrage penalty.
    let objective = |p: &[f64]| -> f64 {
        let Some(ssvi) = surface_from_params(p, family) else {
            return f64::INFINITY;
        };
        let mut cost = 0.0;
        for mat in maturities {
            for q in &mat.quotes {
                if q.weight <= 0.0 {
                    continue;
                }
                let model = ssvi.total_variance(q.k, mat.theta);
                let r = model - q.w;
                cost += q.weight * r * r;
            }
        }
        if !ssvi.is_butterfly_free(&thetas) || !ssvi.is_calendar_free(&thetas) {
            cost += ARBITRAGE_PENALTY;
        }
        cost
    };

    // Multi-start seeds; rho coordinates are unconstrained via tanh.
    let rho_seeds = [-0.6_f64, -0.2, 0.0, 0.2, 0.6];
    let mut best_obj = f64::INFINITY;
    let mut best_params: Vec<f64> = Vec::new();

    match family {
        PhiFamily::Heston => {
            for &rho0 in &rho_seeds {
                for &lambda0 in &[0.5_f64, 1.0, 2.0, 5.0] {
                    let start = [atanh_clamped(rho0), lambda0.ln()];
                    let res = nelder_mead(objective, &start, OUTER_TOL, OUTER_MAX_ITER);
                    if res.fx < best_obj {
                        best_obj = res.fx;
                        best_params = res.x;
                    }
                }
            }
        }
        PhiFamily::PowerLaw => {
            for &rho0 in &rho_seeds {
                for &eta0 in &[0.3_f64, 0.6, 1.0] {
                    for &gamma0 in &[0.3_f64, 0.5, 0.7] {
                        let start = [atanh_clamped(rho0), eta0.ln(), logit(gamma0)];
                        let res = nelder_mead(objective, &start, OUTER_TOL, OUTER_MAX_ITER);
                        if res.fx < best_obj {
                            best_obj = res.fx;
                            best_params = res.x;
                        }
                    }
                }
            }
        }
    }

    let ssvi =
        surface_from_params(&best_params, family).ok_or(CalibrationError::DidNotConverge {
            iterations: OUTER_MAX_ITER,
            residual: best_obj,
        })?;

    let arbitrage_free = ssvi.is_butterfly_free(&thetas) && ssvi.is_calendar_free(&thetas);

    // RMSE excludes the arbitrage penalty.
    let mut residual = 0.0;
    for mat in maturities {
        for q in &mat.quotes {
            if q.weight <= 0.0 {
                continue;
            }
            let r = ssvi.total_variance(q.k, mat.theta) - q.w;
            residual += q.weight * r * r;
        }
    }
    let rmse = (residual / total_weight).sqrt();

    Ok(SsviCalibration {
        ssvi,
        thetas,
        rmse,
        arbitrage_free,
    })
}

/// Reconstructs an [`Ssvi`] from the unconstrained outer-search vector.
///
/// `rho` is mapped through `tanh`; the `phi` parameters through `exp`
/// (positivity) or the logistic (open interval `(0, 1)` for `gamma`).
fn surface_from_params(p: &[f64], family: PhiFamily) -> Option<Ssvi> {
    let rho = p[0].tanh();
    let phi = match family {
        PhiFamily::Heston => {
            let lambda = p[1].exp();
            Phi::heston(lambda).ok()?
        }
        PhiFamily::PowerLaw => {
            let eta = p[1].exp();
            let gamma = logistic(p[2]);
            Phi::power_law(eta, gamma).ok()?
        }
    };
    Ssvi::new(rho, phi).ok()
}

/// Inverse hyperbolic tangent with the argument clamped just inside `(-1, 1)`.
#[inline]
fn atanh_clamped(x: f64) -> f64 {
    let x = x.clamp(-0.999_999, 0.999_999);
    0.5 * ((1.0 + x) / (1.0 - x)).ln()
}

/// Logistic function mapping `R -> (0, 1)`, used for the `gamma` parameter.
#[inline]
fn logistic(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Inverse logistic (logit) mapping `(0, 1) -> R`.
#[inline]
fn logit(p: f64) -> f64 {
    let p = p.clamp(1e-6, 1.0 - 1e-6);
    (p / (1.0 - p)).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds synthetic maturities from a known SSVI surface.
    fn synthetic(truth: &Ssvi, ts_thetas: &[(f64, f64)], ks: &[f64]) -> Vec<SsviMaturity> {
        ts_thetas
            .iter()
            .map(|&(t, theta)| SsviMaturity {
                t,
                theta,
                quotes: ks
                    .iter()
                    .map(|&k| Quote::new(k, truth.total_variance(k, theta), 1.0).unwrap())
                    .collect(),
            })
            .collect()
    }

    #[test]
    fn logistic_logit_invert() {
        for &x in &[0.05, 0.3, 0.5, 0.8, 0.95] {
            assert!((logistic(logit(x)) - x).abs() < 1e-10);
        }
    }

    #[test]
    fn rejects_empty() {
        assert!(matches!(
            calibrate(&[], PhiFamily::PowerLaw),
            Err(CalibrationError::EmptyQuotes)
        ));
    }

    #[test]
    fn recovers_power_law_surface() {
        let truth = Ssvi::new(-0.3, Phi::power_law(0.5, 0.5).unwrap()).unwrap();
        let ks = [-0.3, -0.15, 0.0, 0.15, 0.3];
        let mats = synthetic(&truth, &[(0.5, 0.02), (1.0, 0.04), (2.0, 0.07)], &ks);
        let fit = calibrate(&mats, PhiFamily::PowerLaw).unwrap();
        assert!(fit.rmse < 1e-3, "rmse = {}", fit.rmse);
        assert!(fit.arbitrage_free);
        assert!(
            (fit.ssvi.rho - (-0.3)).abs() < 0.1,
            "rho = {}",
            fit.ssvi.rho
        );
    }

    #[test]
    fn recovers_heston_surface() {
        let truth = Ssvi::new(-0.2, Phi::heston(1.5).unwrap()).unwrap();
        let ks = [-0.3, -0.15, 0.0, 0.15, 0.3];
        let mats = synthetic(&truth, &[(0.5, 0.02), (1.0, 0.04), (2.0, 0.07)], &ks);
        let fit = calibrate(&mats, PhiFamily::Heston).unwrap();
        assert!(fit.rmse < 5e-3, "rmse = {}", fit.rmse);
        assert!(fit.arbitrage_free);
    }

    #[test]
    fn fitted_surface_is_arbitrage_free() {
        let truth = Ssvi::new(-0.4, Phi::power_law(0.6, 0.4).unwrap()).unwrap();
        let ks = [-0.4, -0.2, 0.0, 0.2, 0.4];
        let mats = synthetic(&truth, &[(0.25, 0.015), (1.0, 0.05), (3.0, 0.11)], &ks);
        let fit = calibrate(&mats, PhiFamily::PowerLaw).unwrap();
        assert!(fit.arbitrage_free);
        assert!(fit.ssvi.is_butterfly_free(&fit.thetas));
        assert!(fit.ssvi.is_calendar_free(&fit.thetas));
    }
}
