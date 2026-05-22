// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Direct least-squares slice calibration by Levenberg-Marquardt.
//!
//! A direct calibrator, provided as an alternative to and a refinement step
//! for the quasi-explicit method. It minimises
//!
//! ```text
//! F(chi_R) = sum_i weight_i * ( w_SVI(k_i; chi_R) - w_i )^2
//! ```
//!
//! over `chi_R = {a, b, rho, m, sigma}` with the Levenberg-Marquardt
//! algorithm. The five partial derivatives needed for the Jacobian are
//! closed-form (`u = k - m`, `r = sqrt(u^2 + sigma^2)`):
//!
//! ```text
//! dw/da     = 1
//! dw/db     = rho*u + r
//! dw/drho   = b*u
//! dw/dm     = b*( -rho - u/r )
//! dw/dsigma = b*sigma / r
//! ```
//!
//! No finite differences are used. Domain constraints (`b >= 0`, `|rho| < 1`,
//! `sigma > 0`) are imposed by smooth reparametrisation:
//! `b = exp(b_hat)`, `sigma = exp(sigma_hat)`, `rho = tanh(rho_hat)`. The
//! chain rule carries the Jacobian into the unconstrained coordinates.
//!
//! A good initial guess matters for direct LM; the default seed is the
//! quasi-explicit solution (see [`crate::calibration::quasi_explicit`]),
//! making the two calibrators complementary.
//!
//! # References
//!
//! - Levenberg, K., "A method for the solution of certain non-linear problems
//!   in least squares", *Quarterly of Applied Mathematics* 2(2):164-168 (1944).
//! - Marquardt, D. W., "An algorithm for least-squares estimation of nonlinear
//!   parameters", *Journal of SIAM* 11(2):431-441 (1963).

use crate::calibration::CalibrationResult;
use crate::errors::CalibrationError;
use crate::math::levenberg_marquardt;
use crate::raw::RawSvi;
use crate::types::Quote;

/// Minimum number of quotes the five-parameter raw SVI model can be fit to.
const MIN_QUOTES: usize = 5;
/// Levenberg-Marquardt convergence tolerance.
const LM_TOL: f64 = 1e-14;
/// Levenberg-Marquardt iteration cap.
const LM_MAX_ITER: usize = 500;

/// Refines a raw SVI slice against market quotes by Levenberg-Marquardt.
///
/// Starts the LM iteration from `seed` and returns the locally optimal slice
/// with its RMSE. Best used to polish a quasi-explicit fit; for a standalone
/// calibration prefer [`calibrate`], which seeds itself.
///
/// # Errors
///
/// - [`CalibrationError::EmptyQuotes`] if `quotes` is empty.
/// - [`CalibrationError::TooFewQuotes`] if fewer than five quotes are given.
/// - [`CalibrationError::AllWeightsZero`] if every fitting weight is zero.
/// - [`CalibrationError::Param`] if the refined parameters are invalid.
///
/// # Examples
///
/// ```
/// use regit_svi::types::Quote;
/// use regit_svi::raw::RawSvi;
/// use regit_svi::calibration::least_squares::refine;
///
/// let truth = RawSvi::new(0.04, 0.4, -0.3, 0.05, 0.15).unwrap();
/// let quotes: Vec<Quote> = [-0.3, -0.15, 0.0, 0.15, 0.3]
///     .iter()
///     .map(|&k| Quote::new(k, truth.total_variance(k), 1.0).unwrap())
///     .collect();
/// // A slightly perturbed seed converges back to the truth.
/// let seed = RawSvi::new(0.045, 0.35, -0.2, 0.0, 0.18).unwrap();
/// let fit = refine(&quotes, &seed).unwrap();
/// assert!(fit.rmse < 1e-6);
/// ```
// The five raw SVI parameters and their Jacobian partials carry their
// canonical single-letter names from MATH.md §11 (a, b, rho, m, sigma; dw/da
// etc.); the lints below would only obscure the formula correspondence.
#[allow(clippy::similar_names, clippy::many_single_char_names)]
pub fn refine(quotes: &[Quote], seed: &RawSvi) -> Result<CalibrationResult, CalibrationError> {
    validate_quotes(quotes)?;

    // Unconstrained coordinates: (a, b_hat, rho_hat, m, sigma_hat).
    let start = [
        seed.a,
        seed.b.max(1e-12).ln(),
        atanh(seed.rho.clamp(-0.999_999, 0.999_999)),
        seed.m,
        seed.sigma.max(1e-12).ln(),
    ];

    let residual = |p: &[f64]| -> Vec<(f64, f64, Vec<f64>)> {
        let a = p[0];
        let b_hat = p[1];
        let rho_hat = p[2];
        let m = p[3];
        let sigma_hat = p[4];

        // Map to constrained parameters.
        let b = b_hat.exp();
        let rho = rho_hat.tanh();
        let sigma = sigma_hat.exp();

        // Derivatives of the reparametrisation.
        let db_dbhat = b; // d(exp)/d = exp
        let drho_drhohat = 1.0 - rho * rho; // d(tanh)/d = 1 - tanh^2
        let dsigma_dshat = sigma;

        quotes
            .iter()
            .filter(|q| q.weight > 0.0)
            .map(|q| {
                let u = q.k - m;
                let r = (u * u + sigma * sigma).sqrt();
                let model = a + b * (rho * u + r);
                let resid = model - q.w;

                // Partials in the natural (a, b, rho, m, sigma) coordinates.
                let dw_da = 1.0;
                let dw_db = rho * u + r;
                let dw_drho = b * u;
                let dw_dm = b * (-rho - u / r);
                let dw_dsigma = b * sigma / r;

                // Chain rule into the unconstrained coordinates.
                let jac = vec![
                    dw_da,
                    dw_db * db_dbhat,
                    dw_drho * drho_drhohat,
                    dw_dm,
                    dw_dsigma * dsigma_dshat,
                ];
                (resid, q.weight, jac)
            })
            .collect()
    };

    let res = levenberg_marquardt(residual, &start, LM_TOL, LM_MAX_ITER);

    let a = res.params[0];
    let b = res.params[1].exp();
    let rho = res.params[2].tanh();
    let m = res.params[3];
    let sigma = res.params[4].exp();

    let total_weight: f64 = quotes.iter().map(|q| q.weight).sum();
    let rmse = if total_weight > 0.0 {
        (res.cost / total_weight).sqrt()
    } else {
        0.0
    };

    let slice = RawSvi::new(a, b, rho, m, sigma).map_err(CalibrationError::Param)?;
    Ok(CalibrationResult::new(slice, rmse))
}

/// Calibrates a raw SVI slice by Levenberg-Marquardt with a heuristic seed.
///
/// Builds a starting point from the quoted data (ATM level, observed skew and
/// curvature) and runs [`refine`]. For the most robust standalone fit prefer
/// [`crate::calibration::quasi_explicit::calibrate`]; this function exists so
/// the LM engine is usable on its own.
///
/// # Errors
///
/// Returns the same [`CalibrationError`] variants as [`refine`].
///
/// # Examples
///
/// ```
/// use regit_svi::types::Quote;
/// use regit_svi::calibration::least_squares::calibrate;
///
/// let quotes = [
///     Quote::new(-0.20, 0.0512, 1.0).unwrap(),
///     Quote::new(-0.10, 0.0432, 1.0).unwrap(),
///     Quote::new( 0.00, 0.0400, 1.0).unwrap(),
///     Quote::new( 0.10, 0.0420, 1.0).unwrap(),
///     Quote::new( 0.20, 0.0480, 1.0).unwrap(),
/// ];
/// let fit = calibrate(&quotes).unwrap();
/// assert!(fit.rmse < 1e-1);
/// ```
pub fn calibrate(quotes: &[Quote]) -> Result<CalibrationResult, CalibrationError> {
    validate_quotes(quotes)?;

    // Heuristic seed: ATM level for a, a modest slope/curvature.
    let w_atm = quotes
        .iter()
        .min_by(|x, y| {
            x.k.abs()
                .partial_cmp(&y.k.abs())
                .unwrap_or(core::cmp::Ordering::Equal)
        })
        .map_or(0.04, |q| q.w);
    let k_span = {
        let lo = quotes.iter().map(|q| q.k).fold(f64::INFINITY, f64::min);
        let hi = quotes.iter().map(|q| q.k).fold(f64::NEG_INFINITY, f64::max);
        (hi - lo).max(1e-3)
    };
    let seed = RawSvi::new(0.5 * w_atm, 0.1, -0.1, 0.0, 0.5 * k_span)
        .unwrap_or(RawSvi::new_unchecked(0.5 * w_atm, 0.1, -0.1, 0.0, 0.1));
    refine(quotes, &seed)
}

/// Validates the quote set shared by [`refine`] and [`calibrate`].
fn validate_quotes(quotes: &[Quote]) -> Result<(), CalibrationError> {
    if quotes.is_empty() {
        return Err(CalibrationError::EmptyQuotes);
    }
    if quotes.len() < MIN_QUOTES {
        return Err(CalibrationError::TooFewQuotes {
            got: quotes.len(),
            need: MIN_QUOTES,
        });
    }
    if quotes.iter().all(|q| q.weight <= 0.0) {
        return Err(CalibrationError::AllWeightsZero);
    }
    Ok(())
}

/// Inverse hyperbolic tangent, used to map `rho in (-1, 1)` to the
/// unconstrained `rho_hat`.
#[inline]
fn atanh(x: f64) -> f64 {
    0.5 * ((1.0 + x) / (1.0 - x)).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Generates a synthetic slice of quotes from known raw SVI parameters.
    fn synthetic(svi: &RawSvi, ks: &[f64]) -> Vec<Quote> {
        ks.iter()
            .map(|&k| Quote::new(k, svi.total_variance(k), 1.0).unwrap())
            .collect()
    }

    #[test]
    fn atanh_inverts_tanh() {
        for &x in &[-0.9, -0.3, 0.0, 0.5, 0.95] {
            assert!((atanh(x).tanh() - x).abs() < 1e-12);
        }
    }

    #[test]
    fn rejects_empty() {
        assert!(matches!(calibrate(&[]), Err(CalibrationError::EmptyQuotes)));
    }

    #[test]
    fn rejects_too_few() {
        let q = Quote::new(0.0, 0.04, 1.0).unwrap();
        assert!(matches!(
            calibrate(&[q, q]),
            Err(CalibrationError::TooFewQuotes { .. })
        ));
    }

    #[test]
    fn refine_recovers_from_perturbed_seed() {
        let truth = RawSvi::new(0.04, 0.4, -0.3, 0.05, 0.15).unwrap();
        let ks = [-0.4, -0.25, -0.1, 0.0, 0.1, 0.25, 0.4];
        let quotes = synthetic(&truth, &ks);
        let seed = RawSvi::new(0.05, 0.3, -0.2, 0.0, 0.18).unwrap();
        let fit = refine(&quotes, &seed).unwrap();
        assert!(fit.rmse < 1e-6, "rmse = {}", fit.rmse);
        for &k in &[-0.5, 0.0, 0.5] {
            let err = (fit.slice.total_variance(k) - truth.total_variance(k)).abs();
            assert!(err < 1e-4, "k = {k}, err = {err}");
        }
    }

    #[test]
    fn calibrate_standalone_fits() {
        let truth = RawSvi::new(0.04, 0.35, -0.25, 0.02, 0.16).unwrap();
        let ks = [-0.4, -0.2, -0.05, 0.05, 0.2, 0.4];
        let quotes = synthetic(&truth, &ks);
        let fit = calibrate(&quotes).unwrap();
        // The heuristic seed converges close to the truth.
        for &k in &[-0.3, 0.0, 0.3] {
            let err = (fit.slice.total_variance(k) - truth.total_variance(k)).abs();
            assert!(err < 1e-2, "k = {k}, err = {err}");
        }
    }

    #[test]
    fn refine_preserves_domain() {
        // The exp/tanh reparametrisation must keep the fitted slice valid.
        let truth = RawSvi::new(0.03, 0.5, 0.6, -0.05, 0.12).unwrap();
        let ks = [-0.3, -0.15, 0.0, 0.15, 0.3, 0.45];
        let quotes = synthetic(&truth, &ks);
        let seed = RawSvi::new(0.04, 0.3, 0.3, 0.0, 0.2).unwrap();
        let fit = refine(&quotes, &seed).unwrap();
        assert!(fit.slice.validate().is_ok());
        assert!(fit.slice.b >= 0.0);
        assert!(fit.slice.rho.abs() < 1.0);
        assert!(fit.slice.sigma > 0.0);
    }
}
