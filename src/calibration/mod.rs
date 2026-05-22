// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Slice and surface calibration.
//!
//! Two complementary slice calibrators and a joint SSVI surface fit:
//!
//! - [`quasi_explicit`] — de Marco & Martini / Zeliade quasi-explicit method.
//!   Robust, with no sensitivity to the starting point: the inner convex
//!   problem is solved in closed form and only a 2-D outer search remains.
//! - [`least_squares`] — direct Levenberg-Marquardt over the five raw
//!   parameters with an analytic Jacobian. Fast local polish from a good seed.
//! - [`ssvi`] — joint SSVI surface calibration with the Theorem 4.1 / 4.2
//!   no-arbitrage conditions enforced throughout.
//!
//! The default slice pipeline is [`calibrate_slice`]: a robust quasi-explicit
//! fit followed by an optional Levenberg-Marquardt polish, keeping whichever
//! result has the lower RMSE.
//!
//! # References
//!
//! - De Marco, S. & Martini, C., Zeliade Systems White Paper ZWP-0005 (2009).
//! - Levenberg (1944); Marquardt (1963).
//! - Gatheral, J. & Jacquier, A., *Quantitative Finance* 14(1):59-71 (2014).

pub mod least_squares;
pub mod quasi_explicit;
pub mod ssvi;

use crate::arbitrage::butterfly_scan;
use crate::errors::CalibrationError;
use crate::raw::RawSvi;
use crate::types::Quote;

/// The outcome of a slice calibration: the fitted slice, its fit quality, and
/// its butterfly-arbitrage status.
///
/// # Examples
///
/// ```
/// use regit_svi::raw::RawSvi;
/// use regit_svi::calibration::CalibrationResult;
///
/// let slice = RawSvi::new(0.04, 0.1, -0.2, 0.0, 0.3).unwrap();
/// let result = CalibrationResult::new(slice, 1e-6);
/// assert!(result.butterfly_free);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CalibrationResult {
    /// The calibrated raw SVI slice.
    pub slice: RawSvi,
    /// Weighted root-mean-square fit residual `sqrt(F / sum_i weight_i)`.
    pub rmse: f64,
    /// `true` if the slice passed the butterfly-arbitrage scan.
    pub butterfly_free: bool,
}

impl CalibrationResult {
    /// Builds a result, running the butterfly scan on the slice over its
    /// default domain.
    ///
    /// # Examples
    ///
    /// ```
    /// use regit_svi::raw::RawSvi;
    /// use regit_svi::calibration::CalibrationResult;
    ///
    /// let slice = RawSvi::new(0.04, 0.1, -0.2, 0.0, 0.3).unwrap();
    /// let result = CalibrationResult::new(slice, 1e-6);
    /// assert!((result.rmse - 1e-6).abs() < 1e-15);
    /// ```
    #[must_use]
    pub fn new(slice: RawSvi, rmse: f64) -> Self {
        let butterfly_free = butterfly_scan(&slice, -1.0, 1.0).is_free;
        Self {
            slice,
            rmse,
            butterfly_free,
        }
    }
}

/// Calibrates a raw SVI slice with the default pipeline.
///
/// Runs the quasi-explicit calibrator for a robust global fit, then attempts
/// a Levenberg-Marquardt polish seeded from that result. Whichever fit has
/// the lower RMSE is returned, so the polish can only help.
///
/// # Errors
///
/// Returns a [`CalibrationError`] if the quote set is empty, too small, all
/// weights are zero, or the recovered parameters are invalid.
///
/// # Examples
///
/// ```
/// use regit_svi::types::Quote;
/// use regit_svi::calibration::calibrate_slice;
///
/// let quotes = [
///     Quote::new(-0.20, 0.0512, 1.0).unwrap(),
///     Quote::new(-0.10, 0.0432, 1.0).unwrap(),
///     Quote::new( 0.00, 0.0400, 1.0).unwrap(),
///     Quote::new( 0.10, 0.0420, 1.0).unwrap(),
///     Quote::new( 0.20, 0.0480, 1.0).unwrap(),
/// ];
/// let fit = calibrate_slice(&quotes).unwrap();
/// assert!(fit.rmse < 1e-2);
/// ```
pub fn calibrate_slice(quotes: &[Quote]) -> Result<CalibrationResult, CalibrationError> {
    let qe = quasi_explicit::calibrate(quotes)?;
    // Attempt an LM polish; keep it only if it strictly improves the RMSE.
    match least_squares::refine(quotes, &qe.slice) {
        Ok(lm) if lm.rmse < qe.rmse => Ok(lm),
        _ => Ok(qe),
    }
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
    fn calibration_result_runs_butterfly_scan() {
        let benign = RawSvi::new(0.04, 0.1, -0.2, 0.0, 0.3).unwrap();
        assert!(CalibrationResult::new(benign, 1e-6).butterfly_free);
        let vogt = RawSvi::new(-0.0410, 0.1331, 0.3060, 0.3586, 0.4153).unwrap();
        assert!(!CalibrationResult::new(vogt, 1e-6).butterfly_free);
    }

    #[test]
    fn calibrate_slice_recovers_synthetic() {
        let truth = RawSvi::new(0.04, 0.4, -0.3, 0.05, 0.15).unwrap();
        let ks = [-0.4, -0.25, -0.1, 0.0, 0.1, 0.25, 0.4];
        let quotes = synthetic(&truth, &ks);
        let fit = calibrate_slice(&quotes).unwrap();
        assert!(fit.rmse < 1e-5, "rmse = {}", fit.rmse);
        for &k in &[-0.5, 0.0, 0.5] {
            let err = (fit.slice.total_variance(k) - truth.total_variance(k)).abs();
            assert!(err < 1e-4, "k = {k}, err = {err}");
        }
    }

    #[test]
    fn calibrate_slice_polish_never_worsens() {
        // The pipeline returns the better of the two fits.
        let truth = RawSvi::new(0.03, 0.3, 0.0, 0.0, 0.2).unwrap();
        let ks = [-0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5];
        let quotes = synthetic(&truth, &ks);
        let qe = quasi_explicit::calibrate(&quotes).unwrap();
        let pipeline = calibrate_slice(&quotes).unwrap();
        assert!(pipeline.rmse <= qe.rmse + 1e-12);
    }

    #[test]
    fn calibrate_slice_propagates_errors() {
        assert!(matches!(
            calibrate_slice(&[]),
            Err(CalibrationError::EmptyQuotes)
        ));
    }
}
