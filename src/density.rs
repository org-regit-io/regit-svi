// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Risk-neutral density implied by a raw SVI slice.
//!
//! The butterfly function `g(k)` (see [`crate::arbitrage`]) is exactly the
//! normalised risk-neutral probability density in log-strike space. With
//!
//! ```text
//! d_minus(k) = -k/sqrt(w(k)) - sqrt(w(k))/2
//! d_plus(k)  = -k/sqrt(w(k)) + sqrt(w(k))/2
//! ```
//!
//! the risk-neutral density of the log-strike is
//!
//! ```text
//! p(k) = g(k) / sqrt(2*pi*w(k)) * exp(-d_minus(k)^2 / 2)
//! ```
//!
//! `p(k) >= 0` for all `k` is equivalent to `g(k) >= 0`, which is why the
//! butterfly check is a density-positivity check. For an arbitrage-free slice
//! `p` integrates to 1 over `k in R`; the [`integral`] function provides a
//! numerical check.
//!
//! # References
//!
//! - Breeden, D. & Litzenberger, R., "Prices of state-contingent claims
//!   implicit in option prices", *Journal of Business* 51(4):621-651 (1978).
//! - Gatheral, J. & Jacquier, A., "Arbitrage-free SVI volatility surfaces",
//!   *Quantitative Finance* 14(1):59-71 (2014), eq. (2.2).

use crate::arbitrage::g;
use crate::math::index_to_f64;
use crate::raw::RawSvi;

/// `2*pi` — normalising constant for the density.
const TWO_PI: f64 = std::f64::consts::TAU;

/// The Black `d_minus` quantity: `d_minus(k) = -k/sqrt(w) - sqrt(w)/2`.
///
/// # Examples
///
/// ```
/// use regit_svi::raw::RawSvi;
/// use regit_svi::density::d_minus;
///
/// // At k = 0 with total variance w, d_minus = -sqrt(w)/2.
/// let svi = RawSvi::new(0.04, 0.0, 0.0, 0.0, 0.1).unwrap();
/// assert!((d_minus(&svi, 0.0) + 0.1).abs() < 1e-12);
/// ```
#[must_use]
#[inline]
pub fn d_minus(svi: &RawSvi, k: f64) -> f64 {
    let w = svi.total_variance(k);
    let sqrt_w = w.sqrt();
    -k / sqrt_w - sqrt_w / 2.0
}

/// The Black `d_plus` quantity: `d_plus(k) = -k/sqrt(w) + sqrt(w)/2`.
///
/// # Examples
///
/// ```
/// use regit_svi::raw::RawSvi;
/// use regit_svi::density::d_plus;
///
/// // At k = 0 with total variance w, d_plus = +sqrt(w)/2.
/// let svi = RawSvi::new(0.04, 0.0, 0.0, 0.0, 0.1).unwrap();
/// assert!((d_plus(&svi, 0.0) - 0.1).abs() < 1e-12);
/// ```
#[must_use]
#[inline]
pub fn d_plus(svi: &RawSvi, k: f64) -> f64 {
    let w = svi.total_variance(k);
    let sqrt_w = w.sqrt();
    -k / sqrt_w + sqrt_w / 2.0
}

/// The risk-neutral density `p(k)` implied by a raw SVI slice (MATH.md §9).
///
/// ```text
/// p(k) = g(k) / sqrt(2*pi*w(k)) * exp(-d_minus(k)^2 / 2)
/// ```
///
/// `p(k)` is non-negative everywhere iff the slice is free of butterfly
/// arbitrage. For a slice with butterfly arbitrage `p` is negative on the
/// arbitrage interval.
///
/// # Examples
///
/// ```
/// use regit_svi::raw::RawSvi;
/// use regit_svi::density::risk_neutral_density;
///
/// // A benign slice has a positive density at the money.
/// let svi = RawSvi::new(0.04, 0.1, -0.2, 0.0, 0.3).unwrap();
/// assert!(risk_neutral_density(&svi, 0.0) > 0.0);
/// ```
#[must_use]
pub fn risk_neutral_density(svi: &RawSvi, k: f64) -> f64 {
    let w = svi.total_variance(k);
    if w <= 0.0 || !w.is_finite() {
        return 0.0;
    }
    let dm = d_minus(svi, k);
    g(svi, k) / (TWO_PI * w).sqrt() * (-0.5 * dm * dm).exp()
}

/// A diagnostic report on the risk-neutral density of a raw SVI slice.
///
/// The [`integral`] over a wide log-moneyness window should be close to `1`
/// for an arbitrage-free slice; a value far from `1`, or `min_density < 0`,
/// signals butterfly arbitrage or a numerically extreme slice.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DensityReport {
    /// Numerical integral of `p` over the diagnostic window.
    pub integral: f64,
    /// Smallest density value observed on the integration grid.
    pub min_density: f64,
    /// `true` if `min_density >= 0` over the diagnostic window.
    pub is_non_negative: bool,
}

/// Numerically integrates the risk-neutral density over `[k_lo, k_hi]` by the
/// composite Simpson rule with `2n` panels.
///
/// For an arbitrage-free slice and a sufficiently wide window the result is
/// close to `1` (the density is a probability measure). `n` controls
/// accuracy; `n = 1000` is ample for diagnostic use.
///
/// # Examples
///
/// ```
/// use regit_svi::raw::RawSvi;
/// use regit_svi::density::integral;
///
/// // A benign, low-variance slice integrates to near 1 over a wide window.
/// let svi = RawSvi::new(0.04, 0.05, -0.1, 0.0, 0.4).unwrap();
/// let mass = integral(&svi, -6.0, 6.0, 2000);
/// assert!((mass - 1.0).abs() < 1e-2, "mass = {mass}");
/// ```
#[must_use]
pub fn integral(svi: &RawSvi, k_lo: f64, k_hi: f64, n: usize) -> f64 {
    let n = n.max(1);
    let panels = 2 * n;
    let h = (k_hi - k_lo) / index_to_f64(panels);
    let mut sum = risk_neutral_density(svi, k_lo) + risk_neutral_density(svi, k_hi);
    for i in 1..panels {
        let k = h.mul_add(index_to_f64(i), k_lo);
        let weight = if i % 2 == 1 { 4.0 } else { 2.0 };
        sum += weight * risk_neutral_density(svi, k);
    }
    sum * h / 3.0
}

/// Builds a [`DensityReport`] for a slice over `[k_lo, k_hi]`.
///
/// Integrates the density and scans the same grid for the minimum density,
/// so a single call answers both "does it integrate to one?" and "is it
/// non-negative?".
///
/// # Examples
///
/// ```
/// use regit_svi::raw::RawSvi;
/// use regit_svi::density::density_report;
///
/// let svi = RawSvi::new(0.04, 0.05, -0.1, 0.0, 0.4).unwrap();
/// let report = density_report(&svi, -6.0, 6.0, 2000);
/// assert!(report.is_non_negative);
/// assert!((report.integral - 1.0).abs() < 1e-2);
/// ```
#[must_use]
pub fn density_report(svi: &RawSvi, k_lo: f64, k_hi: f64, n: usize) -> DensityReport {
    let n = n.max(1);
    let panels = 2 * n;
    let h = (k_hi - k_lo) / index_to_f64(panels);

    let mut min_density = f64::INFINITY;
    for i in 0..=panels {
        let k = h.mul_add(index_to_f64(i), k_lo);
        let p = risk_neutral_density(svi, k);
        if p < min_density {
            min_density = p;
        }
    }

    DensityReport {
        integral: integral(svi, k_lo, k_hi, n),
        min_density,
        is_non_negative: min_density >= 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn d_plus_d_minus_differ_by_sqrt_w() {
        let svi = RawSvi::new(0.04, 0.2, -0.3, 0.05, 0.12).unwrap();
        for &k in &[-0.5, 0.0, 0.3] {
            let w = svi.total_variance(k);
            assert!((d_plus(&svi, k) - d_minus(&svi, k) - w.sqrt()).abs() < 1e-12);
        }
    }

    #[test]
    fn density_positive_for_benign_slice() {
        let svi = RawSvi::new(0.04, 0.1, -0.2, 0.0, 0.3).unwrap();
        for &k in &[-1.0, -0.3, 0.0, 0.3, 1.0] {
            assert!(risk_neutral_density(&svi, k) > 0.0, "p({k})");
        }
    }

    #[test]
    fn density_integrates_to_one() {
        let svi = RawSvi::new(0.04, 0.05, -0.1, 0.0, 0.4).unwrap();
        let mass = integral(&svi, -8.0, 8.0, 4000);
        assert!((mass - 1.0).abs() < 1e-3, "mass = {mass}");
    }

    #[test]
    fn density_integrates_to_one_low_vol() {
        let svi = RawSvi::new(0.02, 0.04, -0.15, 0.0, 0.3).unwrap();
        let mass = integral(&svi, -6.0, 6.0, 4000);
        assert!((mass - 1.0).abs() < 1e-3, "mass = {mass}");
    }

    #[test]
    fn density_report_benign_slice() {
        let svi = RawSvi::new(0.04, 0.05, -0.1, 0.0, 0.4).unwrap();
        let report = density_report(&svi, -8.0, 8.0, 4000);
        assert!(report.is_non_negative);
        assert!((report.integral - 1.0).abs() < 1e-3);
        assert!(report.min_density >= 0.0);
    }

    #[test]
    fn density_report_flags_vogt_slice() {
        // The Vogt slice has butterfly arbitrage -> negative density region.
        let vogt = RawSvi::new(-0.0410, 0.1331, 0.3060, 0.3586, 0.4153).unwrap();
        let report = density_report(&vogt, -2.0, 2.0, 2000);
        assert!(!report.is_non_negative);
        assert!(report.min_density < 0.0);
    }

    #[test]
    fn density_handles_zero_variance_gracefully() {
        // A slice whose w_min is exactly 0 should not produce NaN.
        let svi = RawSvi::new(0.0, 0.1, 0.0, 0.0, 0.2).unwrap();
        let p = risk_neutral_density(&svi, svi.k_min());
        assert!(p.is_finite());
    }
}
